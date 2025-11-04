import { Container, Box, Typography, LinearProgress, Button, Stack, Paper } from '@mui/material'
import { useMemo, useState } from 'react'
import axios from 'axios'
import UploadCard from './components/UploadCard'
import VideoPreview from './components/VideoPreview'
import LoadingOverlay from './components/LoadingOverlay'
import Dashboard from './components/Dashboard'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadedName, setUploadedName] = useState<string | null>(null)
  const [processing, setProcessing] = useState(false)
  const [progressMsg, setProgressMsg] = useState('')
  const [outputName, setOutputName] = useState<string | null>(null)
  const [metricsPath, setMetricsPath] = useState<string | null>(null)
  const [metrics, setMetrics] = useState<any>(null)

  const canStart = useMemo(() => !!uploadedName && !processing, [uploadedName, processing])

  const onUpload = async (file: File) => {
    setSelectedFile(file)
    setProcessing(true)
    setProgressMsg('Uploading...')
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await axios.post(`${API_BASE}/upload`, form, { headers: { 'Content-Type': 'multipart/form-data' } })
      setUploadedName(res.data.filename)
    } catch (e: any) {
      alert(e?.response?.data?.detail || 'Upload failed')
    } finally {
      setProcessing(false)
      setProgressMsg('')
    }
  }

  const startProcessing = async () => {
    if (!uploadedName) return
    setProcessing(true)
    setProgressMsg('Analyzing Video...')
    try {
      const form = new FormData()
      form.append('filename', uploadedName)
      const res = await axios.post(`${API_BASE}/process`, form)
      setOutputName(res.data.output)
      if (res.data.metrics) {
        // backend now returns just the metrics filename (basename)
        setMetricsPath(res.data.metrics)
        const metricsRes = await axios.get(`${API_BASE}/download/metrics/${res.data.metrics}`)
        setMetrics(metricsRes.data)
      } else {
        setMetrics(null)
      }
    } catch (e: any) {
      alert(e?.response?.data?.detail || 'Processing failed')
    } finally {
      setProcessing(false)
      setProgressMsg('')
    }
  }

  const download = async () => {
    if (!outputName) return
    window.location.href = `${API_BASE}/download/video/${outputName}`
  }

  return (
    <Container maxWidth="md" sx={{ py: 6 }}>
      <Typography variant="h4" align="center" gutterBottom>
        Drone Detection & Tracking
      </Typography>
      <Typography variant="subtitle1" align="center" color="text.secondary" gutterBottom>
        Upload a video and start detection. We'll render tracking results and summary metrics.
      </Typography>

      <Box sx={{ my: 4 }}>
        <UploadCard onUpload={onUpload} disabled={processing} />
      </Box>

      <Stack direction="row" spacing={2} justifyContent="center" alignItems="center">
        <Button variant="contained" disabled={!canStart} onClick={startProcessing}>
          Start Detection
        </Button>
      </Stack>

      <Box sx={{ my: 4 }}>
        <VideoPreview
          src={outputName ? `${API_BASE}/download/video/${outputName}` : null}
        />
      </Box>

      {metrics && (
        <Box sx={{ my: 2 }}>
          <Dashboard metrics={metrics} />
        </Box>
      )}

      <Stack direction="row" spacing={2} justifyContent="center" sx={{ mt: 2 }}>
        <Button variant="outlined" disabled={!outputName} onClick={download}>
          Download Result
        </Button>
      </Stack>

      <LoadingOverlay open={processing} message={progressMsg} />
    </Container>
  )
}


