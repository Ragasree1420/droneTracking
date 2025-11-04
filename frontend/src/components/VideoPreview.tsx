import { Paper, Typography } from '@mui/material'

export default function VideoPreview({ src }: { src: string | null }) {
  if (!src) return (
    <Paper variant="outlined" sx={{ p: 3, textAlign: 'center' }}>
      <Typography variant="body1" color="text.secondary">No result yet</Typography>
    </Paper>
  )
  return (
    <Paper variant="outlined" sx={{ p: 2 }}>
      <video src={src} controls style={{ width: '100%', maxHeight: 500 }} />
    </Paper>
  )
}


