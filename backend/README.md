# Photo to Video Processor Backend

FastAPI backend service that processes photos from MentraOS glasses into videos with AI-generated music.

## Features

- Photo upload and processing
- Claude AI analysis for music prompt generation
- Suno AI music generation
- Static video creation from photos
- Music-video merging
- Library management for processed videos

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables:
```bash
export SUNO_API_KEY="your_suno_api_key"
export CLAUDE_API_KEY="your_claude_api_key"
export SUNO_BASE_URL="https://studio-api.prod.suno.com/api/v2/external/hackmit"
export TEMP_DIR="temp"
export MAX_FILE_SIZE="10485760"  # 10MB
```

3. Run the server:
```bash
python main.py
```

## API Endpoints

### Photo Processing
- `POST /upload-photo` - Upload photo from glasses
- `GET /status/{task_id}` - Get processing status
- `GET /download/{task_id}` - Download processed video

### Library Management
- `GET /library` - Get all processed videos
- `GET /library/{task_id}` - Get specific video info

### Health Check
- `GET /` - Service status

## Processing Flow

1. Photo uploaded via `/upload-photo`
2. Photo saved to `temp/photos/`
3. Claude AI analyzes photo and generates lyrics
4. Suno AI generates music from lyrics
5. Static video created from photo (30 seconds)
6. Music trimmed to match video duration
7. Video and music merged
8. Final video saved to `temp/output/`

## Configuration

- `MAX_CONCURRENT_TASKS`: Maximum concurrent processing tasks (default: 3)
- `MAX_FILE_SIZE`: Maximum photo file size in bytes (default: 10MB)
- `TEMP_DIR`: Directory for temporary files (default: "temp")

## File Structure

```
temp/
├── photos/     # Uploaded photos
├── music/      # Generated music files
└── output/     # Final processed videos
```