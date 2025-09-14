# AuraTune Backend Service

FastAPI backend service that processes photos from MentraOS glasses into cinematic music videos using AI-generated soundtracks.

## 🎯 Overview

The backend service is the core processing engine of AuraTune. It receives photos from MentraOS glasses, analyzes them with Claude AI, generates custom music with Suno AI, and creates final video outputs.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Suno AI API key
- Claude AI API key
- FFmpeg installed on your system

### Installation

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install FFmpeg** (required for video processing):
   ```bash
   # macOS
   brew install ffmpeg
   
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

### Configuration

1. **Set environment variables**:
   ```bash
   export SUNO_API_KEY="your_suno_api_key"
   export CLAUDE_API_KEY="your_claude_api_key"
   export SUNO_BASE_URL="https://studio-api.prod.suno.com/api/v2/external/hackmit"
   export TEMP_DIR="temp"
   export MAX_FILE_SIZE="10485760"  # 10MB
   ```

2. **Create temp directories** (if they don't exist):
   ```bash
   mkdir -p temp/photos temp/music temp/output
   ```

### Running the Service

1. **Start the backend**:
   ```bash
   python main.py
   ```

2. **Verify it's running**:
   - Backend will be available at `http://localhost:8000`
   - API docs available at `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/`

## 🔧 API Endpoints

### Photo Processing
- `POST /upload-photo` - Upload photo from MentraOS glasses
- `GET /photos/{task_id}/status` - Get processing status
- `GET /photos/{task_id}/download` - Download processed video

### Video Library
- `GET /photos` - Get all processed videos with metadata
- `GET /photos/{task_id}` - Get specific video information

### Health & Status
- `GET /` - Service health check
- `GET /health` - Detailed service status

## 🎵 Processing Pipeline

### Step-by-Step Flow

1. **Photo Upload** (`POST /upload-photo`)
   - Receives base64-encoded photo from MentraOS app
   - Validates file size and format
   - Generates unique task ID
   - Saves photo to `temp/photos/`

2. **AI Analysis** (Claude AI)
   - Analyzes photo content and context
   - Generates descriptive song lyrics
   - Creates music prompt for Suno AI

3. **Music Generation** (Suno AI)
   - Sends lyrics to Suno AI API
   - Generates custom soundtrack
   - Downloads audio file to `temp/music/`

4. **Video Creation** (FFmpeg)
   - Creates 30-second static video from photo
   - Applies cinematic effects and transitions
   - Saves video to `temp/output/`

5. **Audio Merging** (FFmpeg)
   - Trims music to match video duration
   - Merges audio with video
   - Creates final MP4 output

6. **Cleanup**
   - Removes temporary files
   - Updates task status to "done"
   - Makes video available for download

### Concurrent Processing

- **Max concurrent tasks**: 3 (configurable)
- **Rate limiting**: Prevents API overload
- **Error handling**: Robust retry mechanisms
- **Status tracking**: Real-time progress updates

## 📁 File Structure

```
backend/
├── main.py                 # Main FastAPI application
├── requirements.txt        # Python dependencies
├── railway.json           # Railway deployment config
├── railway.env.example    # Environment variables template
├── Dockerfile             # Docker container config
├── temp/                  # Temporary files directory
│   ├── photos/           # Uploaded photos from glasses
│   ├── music/            # Generated music files
│   └── output/           # Final processed videos
└── test_data/            # Test files and outputs
    ├── test_api.py       # API testing script
    └── test_outputs/     # Test video outputs
```

## ⚙️ Configuration Options

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `SUNO_API_KEY` | Suno AI API key | - | ✅ |
| `CLAUDE_API_KEY` | Claude AI API key | - | ✅ |
| `SUNO_BASE_URL` | Suno API endpoint | `https://studio-api.prod.suno.com/api/v2/external/hackmit` | ✅ |
| `TEMP_DIR` | Temporary files directory | `temp` | ❌ |
| `MAX_FILE_SIZE` | Max photo size in bytes | `10485760` (10MB) | ❌ |
| `MAX_CONCURRENT_TASKS` | Max concurrent processing | `3` | ❌ |

### Processing Settings

- **Video duration**: 30 seconds
- **Video format**: MP4 (H.264)
- **Audio format**: MP3
- **Resolution**: Maintains original photo aspect ratio
- **Quality**: High (1080p max)

## 🚨 Troubleshooting

### Common Issues

1. **FFmpeg not found**:
   ```bash
   # Check if FFmpeg is installed
   ffmpeg -version
   
   # If not installed, install it:
   brew install ffmpeg  # macOS
   sudo apt install ffmpeg  # Ubuntu
   ```

2. **API key errors**:
   - Verify API keys are correct and active
   - Check API quotas and limits
   - Ensure proper environment variable loading

3. **File permission errors**:
   ```bash
   # Fix temp directory permissions
   chmod -R 755 temp/
   ```

4. **Memory issues**:
   - Reduce `MAX_CONCURRENT_TASKS`
   - Increase system memory
   - Monitor disk space in temp directory

### Debugging

1. **Enable debug logging**:
   ```bash
   export LOG_LEVEL=DEBUG
   python main.py
   ```

2. **Check processing logs**:
   - Monitor terminal output for errors
   - Check temp directory for intermediate files
   - Verify API responses in network tab

3. **Test individual components**:
   ```bash
   # Test FFmpeg
   ffmpeg -f lavfi -i testsrc=duration=1:size=320x240:rate=1 test.mp4
   
   # Test API endpoints
   curl http://localhost:8000/
   ```

## 📊 Performance Monitoring

### Metrics to Monitor

- **Processing time per video**: ~2-3 minutes average
- **Memory usage**: ~500MB per concurrent task
- **Disk usage**: Monitor temp directory size
- **API response times**: Track external API calls
- **Error rates**: Monitor failed processing attempts

### Optimization Tips

1. **Increase concurrent tasks** for better throughput
2. **Use SSD storage** for temp directory
3. **Monitor API quotas** to avoid rate limiting
4. **Clean up old temp files** regularly

## 🔒 Security Considerations

- **File validation**: Strict file type and size checks
- **Input sanitization**: Clean all user inputs
- **CORS protection**: Configured for specific origins
- **Rate limiting**: Prevents abuse
- **Error handling**: No sensitive data in error messages

## 🚀 Deployment

### Local Development
```bash
python main.py
```

### Production (Railway)
```bash
# Deploy using Railway CLI
railway login
railway link
railway up
```

### Docker
```bash
# Build and run with Docker
docker build -t auratune-backend .
docker run -p 8000:8000 auratune-backend
```

## 📈 Scaling

### Horizontal Scaling
- Deploy multiple backend instances
- Use load balancer for distribution
- Implement shared storage for temp files

### Vertical Scaling
- Increase server memory and CPU
- Use faster storage (SSD)
- Optimize FFmpeg settings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.