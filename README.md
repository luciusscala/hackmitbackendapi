# Mentra + Suno HackMIT Backend

A FastAPI backend that processes video from Mentra smart glasses and generates synchronized music using Suno AI.

## Overview

This system accepts FLV video uploads from RTMP servers, analyzes the video content using Claude AI, generates contextual music with Suno AI, and produces a final MP4 with the generated audio replacing the original.

## Core Functionality

### Video Processing
- Accepts FLV video uploads from RTMP streams
- Converts FLV to MP4 for compatibility
- Extracts video metadata (duration, resolution, frame rate)
- Processes video frames for content analysis

### AI Integration
- **Claude AI**: Analyzes video frames to generate contextual music prompts
- **Suno AI**: Generates music based on video content and style
- Real-time polling for music generation status

### Audio Processing
- Downloads generated music from Suno
- Trims music to match video duration
- Replaces original video audio with generated music
- Preserves music files for debugging

### Concurrency Control
- Maximum 3 concurrent processing tasks
- Async FFmpeg operations to maintain server responsiveness
- Background task processing with status tracking

## API Endpoints

### POST /upload
Upload FLV video file from RTMP server.
- **Input**: FLV file (max 100MB)
- **Output**: Task ID for tracking
- **Response**: `{"task_id": "uuid", "status": "uploaded"}`

### POST /generate/{task_id}
Start music generation process for uploaded video.
- **Input**: Task ID from upload
- **Output**: Background processing initiated
- **Response**: `{"message": "Music generation started"}`

### GET /status/{task_id}
Check processing status and progress.
- **Input**: Task ID
- **Output**: Current status and progress percentage
- **Statuses**: `uploaded`, `converting`, `generating`, `merging`, `done`, `error`

### GET /download/{task_id}
Download final processed video.
- **Input**: Task ID
- **Output**: MP4 file with generated music
- **Note**: Cleans up temporary files after download

## Processing Pipeline

1. **Upload**: Store FLV video temporarily
2. **Convert**: Convert FLV to MP4 using FFmpeg
3. **Analyze**: Extract metadata and analyze video frames
4. **Generate**: Create music prompt and call Suno API
5. **Poll**: Wait for music generation completion
6. **Download**: Retrieve generated MP3 from Suno
7. **Trim**: Match music duration to video length
8. **Merge**: Replace video audio with generated music
9. **Complete**: Return final MP4 with new audio

## Setup

### Prerequisites
- Python 3.8+
- FFmpeg installed on system
- Suno API key (HackMIT 2025)
- Claude API key

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
Create `.env` file:
```bash
SUNO_API_KEY=your_suno_api_key_here
SUNO_BASE_URL=https://studio-api.prod.suno.com/api/v2/external/hackmit
CLAUDE_API_KEY=your_claude_api_key_here
TEMP_DIR=temp
MAX_FILE_SIZE=104857600
```

### Run Server
```bash
python main.py
```

## Integration

### RTMP Server Integration
```bash
# Upload video after recording
curl -X POST "http://localhost:8000/upload" -F "file=@video.flv"

# Start processing
curl -X POST "http://localhost:8000/generate/{task_id}"

# Monitor progress
curl "http://localhost:8000/status/{task_id}"

# Download result
curl "http://localhost:8000/download/{task_id}" -o output.mp4
```

### Health Check
```bash
curl "http://localhost:8000/"
```
Returns server status and active task count.

## File Structure

```
hackmit/
├── main.py              # FastAPI application
├── test_api.py          # End-to-end test suite
├── requirements.txt     # Dependencies
├── temp/               # Temporary storage
│   ├── videos/         # Uploaded FLV files
│   ├── music/          # Generated MP3 files
│   └── output/         # Final MP4 files
└── test_data/          # Test video files
```

## Error Handling

- Input validation for file types and sizes
- Timeout handling for external API calls
- Graceful error recovery with detailed messages
- Background task error logging

## Performance

- Async processing maintains server responsiveness
- Concurrent task limiting prevents resource exhaustion
- Efficient FFmpeg operations with proper timeouts
- Memory-optimized file handling