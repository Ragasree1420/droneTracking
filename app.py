from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, Response
import os
import cv2
import sys
import time
import torch
import logging
import datetime
import numpy as np
from PIL import Image
import warnings
import tempfile
import threading
from werkzeug.utils import secure_filename

# Add paths for the detection and tracking modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'yolov5'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'detect_wrapper'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tracking_wrapper', 'dronetracker'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'tracking_wrapper', 'drtracker'))

from detect_wrapper.Detectoruav import DroneDetection
from tracking_wrapper.dronetracker.trackinguav.evaluation.tracker import Tracker

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Global variables for detection and tracking
detector = None
tracker = None
processing_status = {}

def init_models():
    """Initialize the detection and tracking models"""
    global detector, tracker
    
    if detector is None or tracker is None:
        print("Initializing models...")
        
        # Initialize detector
        ir_weights_path = os.path.join(os.path.dirname(__file__), 'detect_wrapper', 'weights', 'best.pt')
        # Try to locate an RGB weights file under user's checkpoints directory; fallback to IR weights if not found
        rgb_dir = os.path.join(os.path.dirname(__file__), 'detect_wrapper', 'weights', 'drone_rgb_yolov5s.pt')
        rgb_weights_path = ir_weights_path
        try:
            if os.path.isdir(rgb_dir):
                candidates = [f for f in os.listdir(rgb_dir) if f.lower().endswith('.pt')]
                if candidates:
                    # Choose the first candidate deterministically (sorted)
                    candidates.sort()
                    rgb_weights_path = os.path.join(rgb_dir, candidates[0])
        except Exception as _e:
            rgb_weights_path = ir_weights_path
        
        detector = DroneDetection(IRweights_path=ir_weights_path, RGBweights_path=rgb_weights_path)
        
        # Initialize tracker
        tracker = Tracker()
        
        print("Models initialized successfully!")

def mono_to_rgb(data):
    """Convert monochrome image to RGB"""
    w, h = data.shape
    img = np.zeros((w, h, 3), dtype=np.uint8)
    img[:, :, 0] = data
    img[:, :, 1] = data
    img[:, :, 2] = data
    return img

def distance_check(bbx1, bbx2, thd):
    """Check if two bounding boxes are close enough"""
    cx1 = bbx1[0] + bbx1[2] / 2
    cy1 = bbx1[1] + bbx1[3] / 2
    cx2 = bbx2[0] + bbx2[2] / 2
    cy2 = bbx2[1] + bbx2[3] / 2
    dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    return dist < thd

def scale_coords(img1_shape, coords, img0_shape):
    """Scale coordinates from one image shape to another"""
    gainx = img1_shape[0] / img0_shape[0]
    gainy = img1_shape[1] / img0_shape[1]
    coords[0] = coords[0] / gainx
    coords[1] = coords[1] / gainy
    coords[2] = coords[2] / gainx
    coords[3] = coords[3] / gainy
    coords = [int(x) for x in coords]
    return coords

def process_video(video_path, output_path, job_id):
    """Process video with detection and tracking"""
    global detector, tracker, processing_status
    
    try:
        processing_status[job_id] = {'status': 'processing', 'progress': 0, 'message': 'Starting video processing...', 'drone_present': False}
        
        # Initialize models if not already done
        init_models()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Processing variables
        frame_counter = 0
        track_counter = 0
        detect_first = True
        TRACK_MAX_COUNT = 150
        count = 0
        
        # Add detection stability variables
        detection_history = []
        stable_detection_count = 0
        no_detection_count = 0
        last_detection_state = None
        DETECTION_THRESHOLD = 3  # Number of consecutive detections needed to confirm
        
        processing_status[job_id]['message'] = f'Processing {total_frames} frames...'
        
        # Decide IR vs RGB using the first readable frame
        def _detect_video_type(sample_frame):
            # If single channel, it's IR
            if len(sample_frame.shape) == 2 or sample_frame.shape[2] == 1:
                return 'IR'
            # Measure colorfulness; low colorfulness often indicates IR-like grayscale
            b, g, r = cv2.split(sample_frame)
            rg = cv2.absdiff(r, g)
            yb = cv2.absdiff(0.5 * (r + g).astype(np.float32), b.astype(np.float32))
            std_rg = np.std(rg)
            std_yb = np.std(yb)
            colorfulness = std_rg + std_yb
            return 'RGB' if colorfulness > 5.0 else 'IR'

        video_mode = None
        
        detected_any = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            count += 1
            frame_counter += 1
            
            # Update progress
            progress = int((frame_counter / total_frames) * 100)
            processing_status[job_id]['progress'] = progress
            
            # Determine image type (IR or RGB) once using the first frame
            if video_mode is None:
                if len(frame.shape) == 2:
                    video_mode = 'IR'
                    frame = mono_to_rgb(frame)
                else:
                    video_mode = _detect_video_type(frame)
            
            bbx = None
            
            if detector and tracker:
                if track_counter <= 0:
                    # Detection phase
                    if video_mode == 'RGB':
                        init_box = detector.forward_RGB(frame)
                        center_box = [320, 192, 0, 0]
                    else:
                        init_box = detector.forward_IR(frame)
                        center_box = [320, 256, 0, 0]
                    
                    # Update detection stability
                    current_detection = init_box is not None
                    detection_history.append(current_detection)
                    if len(detection_history) > DETECTION_THRESHOLD:
                        detection_history.pop(0)
                    
                    # Determine stable detection state
                    if len(detection_history) >= DETECTION_THRESHOLD:
                        stable_detection = sum(detection_history) >= DETECTION_THRESHOLD // 2 + 1
                    else:
                        stable_detection = current_detection
                    
                    # Only change state if we have a stable detection
                    if stable_detection != last_detection_state:
                        if stable_detection:
                            stable_detection_count += 1
                            no_detection_count = 0
                        else:
                            no_detection_count += 1
                            stable_detection_count = 0
                        
                        # Only update state after threshold
                        if stable_detection_count >= DETECTION_THRESHOLD or no_detection_count >= DETECTION_THRESHOLD:
                            last_detection_state = stable_detection
                            stable_detection_count = 0
                            no_detection_count = 0
                    
                    # Use stable detection for drone presence tracking
                    if stable_detection:
                        detected_any = True
                        processing_status[job_id]['drone_present'] = True
                    
                    if detect_first:
                        if init_box is not None and last_detection_state:
                            if distance_check(init_box, center_box, 60) and count % 4 == 1:
                                tracker.init_track(init_box, frame)
                                track_counter = TRACK_MAX_COUNT
                                detect_first = False
                            
                            init_box = [int(x) for x in init_box]
                            bbx = init_box
                    else:
                        if init_box is not None and distance_check(init_box, center_box, 60) and last_detection_state:
                            tracker.change_state(init_box)
                            track_counter = TRACK_MAX_COUNT
                            init_box = [int(x) for x in init_box]
                            bbx = init_box
                        else:
                            # Only track if tracker has been initialized
                            if not detect_first and last_detection_state:
                                track_counter = TRACK_MAX_COUNT
                                bbx = tracker.on_track(frame)
                                track_counter -= 1
                            else:
                                bbx = None
                else:
                    # Tracking phase
                    if last_detection_state:
                        bbx = tracker.on_track(frame)
                        track_counter -= 1
                    else:
                        bbx = None
            
            # Draw bounding box on frame or "No Drone Detected" caption based on stable detection
            if bbx is not None and last_detection_state:
                x, y, w, h = bbx
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, 'Drone', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif not last_detection_state:
                # Add "No Drone Detected" caption when no drone is found
                text = "No Drone Detected"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                color = (0, 0, 255)  # Red color
                thickness = 2
                
                # Get text size for centering
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                frame_height, frame_width = frame.shape[:2]
                
                # Center the text
                x = (frame_width - text_width) // 2
                y = (frame_height + text_height) // 2
                
                # Add background rectangle for better visibility
                padding = 10
                cv2.rectangle(frame, 
                            (x - padding, y - text_height - padding), 
                            (x + text_width + padding, y + baseline + padding), 
                            (255, 255, 255), -1)
                
                # Add the text
                cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
            
            # Write frame to output video
            out.write(frame)
        
        # Cleanup
        cap.release()
        out.release()
        
        processing_status[job_id] = {
            'status': 'completed', 
            'progress': 100, 
            'message': 'Video processing completed successfully!',
            'output_path': output_path,
            'drone_present': detected_any
        }
        
    except Exception as e:
        processing_status[job_id] = {
            'status': 'error', 
            'progress': 0, 
            'message': f'Error processing video: {str(e)}'
        }
        print(f"Error processing video: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Generate unique job ID
        job_id = str(int(time.time()))
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_{filename}")
        file.save(video_path)
        
        # Generate output path
        output_filename = f"processed_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_{output_filename}")
        
        # Start processing in background thread
        thread = threading.Thread(target=process_video, args=(video_path, output_path, job_id))
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'message': 'Video uploaded successfully. Processing started.',
            'status_url': f'/status/{job_id}'
        })
    
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/status/<job_id>')
def get_status(job_id):
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    status = processing_status[job_id]
    return jsonify(status)

@app.route('/download/<job_id>')
def download_result(job_id):
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    status = processing_status[job_id]
    if status['status'] != 'completed':
        return jsonify({'error': 'Video not ready yet'}), 400
    
    output_path = status['output_path']
    if not os.path.exists(output_path):
        return jsonify({'error': 'Output file not found'}), 404
    
    return send_file(output_path, as_attachment=True)

@app.route('/preview/<job_id>')
def preview_result(job_id):
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    status = processing_status[job_id]
    if status['status'] != 'completed':
        return jsonify({'error': 'Video not ready yet'}), 400
    
    output_path = status['output_path']
    if not os.path.exists(output_path):
        return jsonify({'error': 'Output file not found'}), 404

    # Support HTTP Range requests for HTML5 video playback (seekable streaming)
    range_header = request.headers.get('Range', None)
    file_size = os.path.getsize(output_path)
    if range_header:
        # Example Range: bytes=Start-End
        bytes_range = range_header.strip().split('=')[-1]
        start_str, end_str = (bytes_range.split('-') + [None])[:2]
        try:
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
        except ValueError:
            start = 0
            end = file_size - 1
        end = min(end, file_size - 1)
        length = end - start + 1

        def generate():
            with open(output_path, 'rb') as f:
                f.seek(start)
                remaining = length
                chunk_size = 8192
                while remaining > 0:
                    chunk = f.read(min(chunk_size, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        rv = Response(generate(), status=206, mimetype='video/mp4')
        rv.headers.add('Content-Range', f'bytes {start}-{end}/{file_size}')
        rv.headers.add('Accept-Ranges', 'bytes')
        rv.headers.add('Content-Length', str(length))
        return rv

    # Fallback: send whole file
    return send_file(output_path, as_attachment=False, mimetype='video/mp4')

@app.route('/cleanup/<job_id>')
def cleanup_job(job_id):
    """Clean up uploaded and processed files for a job"""
    try:
        # Remove uploaded file
        upload_pattern = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}_*")
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            if file.startswith(f"{job_id}_"):
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
        
        # Remove output file
        output_pattern = os.path.join(app.config['OUTPUT_FOLDER'], f"{job_id}_*")
        for file in os.listdir(app.config['OUTPUT_FOLDER']):
            if file.startswith(f"{job_id}_"):
                os.remove(os.path.join(app.config['OUTPUT_FOLDER'], file))
        
        # Remove from status
        if job_id in processing_status:
            del processing_status[job_id]
        
        return jsonify({'message': 'Files cleaned up successfully'})
    except Exception as e:
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Drone Hunter Web Application...")
    print("Initializing models...")
    init_models()
    print("Models initialized. Starting web server...")
    app.run(debug=True, host='0.0.0.0', port=5000)

