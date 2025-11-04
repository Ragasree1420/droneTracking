import { Container, Box, Typography, Button, Stack, Paper, CssBaseline } from '@mui/material'
import { useMemo, useState } from 'react'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import axios from 'axios'
import UploadCard from './components/UploadCard'
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

  const theme = createTheme({
    palette: {
      mode: 'light',
      background: {
        default: '#e8ecf1'
      },
      primary: {
        main: '#2563eb',
        dark: '#1d4ed8'
      },
      text: {
        primary: '#1e293b'
      }
    },
    typography: {
      fontFamily: 'Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif'
    },
    shape: {
      borderRadius: 16
    },
    components: {
      MuiPaper: {
        styleOverrides: {
          root: {
            borderRadius: '1rem',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.08)',
            backgroundColor: '#ffffff'
          }
        }
      },
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: '0.75rem',
            textTransform: 'none',
            fontWeight: 600
          }
        }
      }
    }
  })

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="md" sx={{ py: 6, minHeight: '100dvh', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
        <Typography variant="h4" align="center" gutterBottom sx={{ color: 'text.primary' }}>
          Drone Detection & Tracking
        </Typography>
        <Typography variant="subtitle1" align="center" color="text.secondary" gutterBottom>
          Upload a video and start detection. We’ll process it and show final results.
        </Typography>

        <Box sx={{ my: 4 }}>
          <UploadCard onUpload={onUpload} disabled={processing} />
        </Box>

        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} justifyContent="center" alignItems="center">
          <Button variant="contained" color="primary" disabled={!canStart} onClick={startProcessing}>
            Start Detection
          </Button>
          <Button variant="outlined" disabled={!outputName} onClick={download}>
            Download Result
          </Button>
        </Stack>

        {/* No video preview while uploading or processing; show only final results */}
        {processing && (
          <Box sx={{ mt: 4 }}>
            <Paper sx={{ p: 3, textAlign: 'center' }}>
              <Typography variant="body1" color="text.secondary">Processing video, please wait…</Typography>
            </Paper>
          </Box>
        )}

        {!processing && outputName && (
          <Box sx={{ mt: 4 }}>
            <Paper sx={{ p: 3, textAlign: 'center' }}>
              <Typography variant="h6" gutterBottom>Processing Complete</Typography>
              <Typography variant="body2" color="text.secondary">
                Your video has been processed. You can download the result and view the summary below.
              </Typography>
            </Paper>
          </Box>
        )}

        {!processing && metrics && (
          <Box sx={{ my: 2 }}>
            <Dashboard metrics={metrics} />
          </Box>
        )}

        <LoadingOverlay open={processing} message={progressMsg || 'Processing video, please wait…'} />
      </Container>
    </ThemeProvider>
  )
}


