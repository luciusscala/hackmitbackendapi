# Mentra + Suno HackMIT Backend

FastAPI backend server that integrates Mentra smart glasses with Suno AI music generation.

## Features

- Accept FLV video uploads from RTMP server
- Convert FLV to MP4 for better compatibility
- Generate music prompts based on video metadata
- Call Suno API for AI music generation
- Merge video and audio using FFmpeg
- Real-time status tracking
- Automatic file cleanup

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
# Create .env file
SUNO_API_KEY=your_suno_api_key_here
SUNO_BASE_URL=https://api.suno.ai
HOST=0.0.0.0
PORT=8000
TEMP_DIR=temp
MAX_FILE_SIZE=104857600
```

3. Run the server:
```bash
python main.py
```

## API Endpoints

### Upload Video
```
POST /upload
- Accept FLV file from RTMP server
- Returns task_id for tracking
```

### Generate Music
```
POST /generate/{task_id}
- Start music generation process
- Runs in background
```

### Check Status
```
GET /status/{task_id}
- Returns current processing status
- Statuses: uploaded, converting, generating, merging, done, error
```

### Download Result
```
GET /download/{task_id}
- Download final merged MP4 video
- Cleans up temp files after download
```

## Processing Pipeline

1. **Upload** → FLV video stored temporarily
2. **Convert** → FLV to MP4 conversion
3. **Analyze** → Extract video metadata (duration, resolution)
4. **Generate** → Create music prompt and call Suno API
5. **Poll** → Wait for music generation completion
6. **Download** → Get generated MP3 from Suno
7. **Merge** → Combine video and audio with FFmpeg
8. **Cleanup** → Remove temporary files

## RTMP Server Integration

Your RTMP server can integrate by making HTTP POST requests:

```bash
# Upload video after recording stops
curl -X POST "http://localhost:8000/upload" \
  -F "file=@recorded_video.flv"

# Start music generation
curl -X POST "http://localhost:8000/generate/{task_id}"

# Check status
curl "http://localhost:8000/status/{task_id}"

# Download final video
curl "http://localhost:8000/download/{task_id}" -o final_video.mp4
```

## Requirements

- Python 3.8+
- FFmpeg installed on system
- Suno API access
- RTMP server for video capture

## File Structure

```
backend/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── .env                # Environment variables
├── temp/               # Temporary file storage
│   ├── videos/         # Uploaded FLV files
│   ├── music/          # Generated MP3 files
│   └── output/         # Final merged videos
└── README.md           # This file
```
