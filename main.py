import os
import uuid
import asyncio
import shutil
from datetime import datetime
from typing import Dict, Optional
from enum import Enum
from pathlib import Path

import httpx
import ffmpeg
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SUNO_API_KEY = os.getenv("SUNO_API_KEY", "your_suno_api_key_here")
SUNO_BASE_URL = os.getenv("SUNO_BASE_URL", "https://studio-api.prod.suno.com/api/v2/external/hackmit")
TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "104857600"))  # 100MB

# Ensure temp directories exist
TEMP_DIR.mkdir(exist_ok=True)
(TEMP_DIR / "videos").mkdir(exist_ok=True)
(TEMP_DIR / "music").mkdir(exist_ok=True)
(TEMP_DIR / "output").mkdir(exist_ok=True)

# FastAPI app
app = FastAPI(title="Mentra + Suno HackMIT Backend", version="1.0.0")

# Data Models
class TaskStatus(str, Enum):
    UPLOADED = "uploaded"
    CONVERTING = "converting"
    GENERATING = "generating"
    MERGING = "merging"
    DONE = "done"
    ERROR = "error"

class Task(BaseModel):
    task_id: str
    status: TaskStatus
    progress: int
    video_path: str
    music_path: Optional[str] = None
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class UploadResponse(BaseModel):
    task_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    message: str
    error_message: Optional[str] = None

# In-memory storage
tasks: Dict[str, Task] = {}

# Helper Functions
def generate_task_id() -> str:
    return str(uuid.uuid4())

def get_video_metadata(video_path: str) -> dict:
    """Extract video metadata using ffmpeg"""
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
        
        return {
            'duration': float(video_stream.get('duration', 0)),
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'fps': eval(video_stream.get('r_frame_rate', '0/1'))
        }
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        return {'duration': 30, 'width': 1920, 'height': 1080, 'fps': 30}

def generate_music_prompt(metadata: dict) -> str:
    """Generate music prompt based on video metadata"""
    duration = int(metadata.get('duration', 30))
    width = metadata.get('width', 1920)
    height = metadata.get('height', 1080)
    
    # Simple prompt generation based on video characteristics
    if duration < 15:
        tempo = "fast-paced"
    elif duration < 60:
        tempo = "upbeat"
    else:
        tempo = "moderate"
    
    if width >= 1920 and height >= 1080:
        quality = "high-quality"
    else:
        quality = "standard"
    
    return f"{tempo} music for {duration}s {quality} video, energetic soundtrack"

async def call_suno_api(prompt: str) -> str:
    """Call Suno API to generate music"""
    headers = {
        "Authorization": f"Bearer {SUNO_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "prompt": prompt,
        "duration": 30,  # Default duration
        "tags": "upbeat, energetic, soundtrack"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{SUNO_BASE_URL}/generate",
                headers=headers,
                json=data,
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()
            return result.get("id", "unknown")
        except Exception as e:
            print(f"Suno API error: {e}")
            raise HTTPException(status_code=500, detail=f"Suno API error: {str(e)}")

async def poll_suno_status(suno_task_id: str) -> Optional[str]:
    """Poll Suno API for music generation status"""
    headers = {
        "Authorization": f"Bearer {SUNO_API_KEY}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{SUNO_BASE_URL}/clips?ids={suno_task_id}",
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            result = response.json()
            
            # The response is an array of clips
            if isinstance(result, list) and len(result) > 0:
                clip = result[0]
                status = clip.get("status")
                
                if status == "completed" or status == "streaming":
                    return clip.get("audio_url")
            return None
        except Exception as e:
            print(f"Suno polling error: {e}")
            return None

async def download_music(audio_url: str, output_path: str):
    """Download generated music from Suno"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(audio_url, timeout=60.0)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
        except Exception as e:
            print(f"Music download error: {e}")
            raise

async def convert_flv_to_mp4(flv_path: str, mp4_path: str):
    """Convert FLV to MP4 using ffmpeg"""
    try:
        (
            ffmpeg
            .input(flv_path)
            .output(mp4_path, vcodec='libx264', acodec='aac')
            .overwrite_output()
            .run(quiet=True)
        )
    except Exception as e:
        print(f"FFmpeg conversion error: {e}")
        raise

async def merge_video_audio(video_path: str, audio_path: str, output_path: str):
    """Merge video and audio using ffmpeg - overlay music on existing video audio"""
    try:
        video = ffmpeg.input(video_path)
        music = ffmpeg.input(audio_path)
        
        # Mix the original video audio with the generated music
        # This overlays the music on top of the existing audio
        mixed_audio = ffmpeg.filter([video['a'], music['a']], 'amix', inputs=2, duration='shortest')
        
        (
            ffmpeg
            .output(video['v'], mixed_audio, output_path, vcodec='copy', acodec='aac')
            .overwrite_output()
            .run(quiet=True)
        )
    except Exception as e:
        print(f"FFmpeg merge error: {e}")
        raise

async def cleanup_temp_files(task_id: str):
    """Clean up temporary files"""
    try:
        task = tasks.get(task_id)
        if not task:
            return
            
        # Remove video files
        if task.video_path and os.path.exists(task.video_path):
            os.remove(task.video_path)
        
        # Remove music files
        if task.music_path and os.path.exists(task.music_path):
            os.remove(task.music_path)
            
    except Exception as e:
        print(f"Cleanup error: {e}")

async def process_video_task(task_id: str):
    """Background task to process video"""
    task = tasks.get(task_id)
    if not task:
        return
    
    try:
        # Step 1: Convert FLV to MP4
        task.status = TaskStatus.CONVERTING
        task.progress = 20
        task.updated_at = datetime.now()
        
        flv_path = task.video_path
        mp4_path = str(TEMP_DIR / "videos" / f"{task_id}.mp4")
        
        await convert_flv_to_mp4(flv_path, mp4_path)
        task.video_path = mp4_path
        
        # Step 2: Analyze video and generate music
        task.status = TaskStatus.GENERATING
        task.progress = 40
        task.updated_at = datetime.now()
        
        metadata = get_video_metadata(mp4_path)
        prompt = generate_music_prompt(metadata)
        
        # Call Suno API
        suno_task_id = await call_suno_api(prompt)
        
        # Poll for completion
        audio_url = None
        max_attempts = 30  # 5 minutes max
        for attempt in range(max_attempts):
            await asyncio.sleep(10)  # Wait 10 seconds between polls
            audio_url = await poll_suno_status(suno_task_id)
            if audio_url:
                break
            task.progress = 40 + (attempt * 2)  # Progress from 40 to 98
        
        if not audio_url:
            raise Exception("Music generation timeout")
        
        # Download music
        music_path = str(TEMP_DIR / "music" / f"{task_id}.mp3")
        await download_music(audio_url, music_path)
        task.music_path = music_path
        
        # Step 3: Merge video and audio
        task.status = TaskStatus.MERGING
        task.progress = 90
        task.updated_at = datetime.now()
        
        output_path = str(TEMP_DIR / "output" / f"{task_id}.mp4")
        await merge_video_audio(mp4_path, music_path, output_path)
        task.output_path = output_path
        
        # Step 4: Complete
        task.status = TaskStatus.DONE
        task.progress = 100
        task.updated_at = datetime.now()
        
    except Exception as e:
        task.status = TaskStatus.ERROR
        task.error_message = str(e)
        task.updated_at = datetime.now()
        print(f"Task {task_id} error: {e}")
        import traceback
        traceback.print_exc()

# API Endpoints
@app.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload FLV video from RTMP server"""
    
    # Validate file
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    if not file.filename.endswith('.flv'):
        raise HTTPException(status_code=400, detail="Only FLV files are supported")
    
    # Generate task ID and save file
    task_id = generate_task_id()
    video_path = str(TEMP_DIR / "videos" / f"{task_id}.flv")
    
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")
    
    # Create task
    task = Task(
        task_id=task_id,
        status=TaskStatus.UPLOADED,
        progress=0,
        video_path=video_path,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    tasks[task_id] = task
    
    return UploadResponse(
        task_id=task_id,
        status="uploaded",
        message="Video uploaded successfully"
    )

@app.post("/generate/{task_id}")
async def generate_music(task_id: str, background_tasks: BackgroundTasks):
    """Start music generation for uploaded video"""
    
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    if task.status != TaskStatus.UPLOADED:
        raise HTTPException(status_code=400, detail="Task already processed or in progress")
    
    # Start background processing
    background_tasks.add_task(process_video_task, task_id)
    
    return {"message": "Music generation started", "task_id": task_id}

@app.get("/status/{task_id}", response_model=StatusResponse)
async def get_status(task_id: str):
    """Get processing status for a task"""
    
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    return StatusResponse(
        task_id=task_id,
        status=task.status.value,
        progress=task.progress,
        message=f"Task is {task.status.value}",
        error_message=task.error_message
    )

@app.get("/download/{task_id}")
async def download_video(task_id: str):
    """Download final merged video"""
    
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    if task.status != TaskStatus.DONE:
        raise HTTPException(status_code=400, detail="Video not ready yet")
    
    if not task.output_path or not os.path.exists(task.output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    # Clean up temp files after download
    asyncio.create_task(cleanup_temp_files(task_id))
    
    return FileResponse(
        path=task.output_path,
        filename=f"merged_video_{task_id}.mp4",
        media_type="video/mp4"
    )

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Mentra + Suno HackMIT Backend",
        "status": "running",
        "active_tasks": len(tasks)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
