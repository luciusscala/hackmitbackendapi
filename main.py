import os
import uuid
import asyncio
import shutil
import base64
import subprocess
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
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# Configuration
SUNO_API_KEY = os.getenv("SUNO_API_KEY", "your_suno_api_key_here")
SUNO_BASE_URL = os.getenv("SUNO_BASE_URL", "https://studio-api.prod.suno.com/api/v2/external/hackmit")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "your_claude_api_key_here")
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

# Concurrency control
MAX_CONCURRENT_TASKS = 3
active_tasks = 0
task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

# Helper Functions
def generate_task_id() -> str:
    return str(uuid.uuid4())

async def run_ffmpeg_async(cmd: list, timeout: int = 300) -> tuple[int, str, str]:
    """Run FFmpeg command asynchronously"""
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), 
            timeout=timeout
        )
        
        return process.returncode, stdout.decode(), stderr.decode()
    except asyncio.TimeoutError:
        process.kill()
        return -1, "", "FFmpeg timeout"
    except Exception as e:
        return -1, "", str(e)

async def get_video_metadata(video_path: str) -> dict:
    """Extract video metadata using ffmpeg asynchronously"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-show_format', video_path
        ]
        
        returncode, stdout, stderr = await run_ffmpeg_async(cmd, timeout=30)
        
        if returncode != 0:
            raise Exception(f"FFprobe failed: {stderr}")
        
        import json
        data = json.loads(stdout)
        video_stream = next(stream for stream in data['streams'] if stream['codec_type'] == 'video')
        
        return {
            'duration': float(video_stream.get('duration', 0)),
            'width': int(video_stream.get('width', 0)),
            'height': int(video_stream.get('height', 0)),
            'fps': eval(video_stream.get('r_frame_rate', '0/1'))
        }
    except Exception as e:
        return {'duration': 30, 'width': 1920, 'height': 1080, 'fps': 30}

async def extract_video_frames(video_path: str, num_frames: int = 3) -> list:
    """Extract key frames from video for analysis asynchronously"""
    try:
        # Get video duration first
        metadata = await get_video_metadata(video_path)
        duration = metadata.get('duration', 30)
        
        frames = []
        
        # Extract frames at different points in the video
        for i in range(num_frames):
            timestamp = (duration / (num_frames + 1)) * (i + 1)
            frame_path = str(TEMP_DIR / f"frame_{i}.jpg")
            
            cmd = [
                'ffmpeg', '-y', '-v', 'quiet',
                '-ss', str(timestamp),
                '-i', video_path,
                '-vframes', '1',
                '-f', 'image2',
                '-vcodec', 'mjpeg',
                frame_path
            ]
            
            returncode, stdout, stderr = await run_ffmpeg_async(cmd, timeout=30)
            
            if returncode == 0 and os.path.exists(frame_path):
                frames.append(frame_path)
        
        return frames
    except Exception as e:
        print(f"Frame extraction error: {e}")
        return []

def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 for Claude API"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Image encoding error: {e}")
        return ""

async def analyze_video_with_claude(video_path: str, metadata: dict) -> str:
    """Use Claude API to analyze video frames and generate music prompt"""
    try:
        # Extract frames from video
        frames = await extract_video_frames(video_path, num_frames=3)
        if not frames:
            raise HTTPException(status_code=500, detail="Failed to extract video frames for analysis")
        
        # Initialize Claude client
        client = Anthropic(api_key=CLAUDE_API_KEY)
        
        # Prepare images for Claude
        image_data = []
        for frame_path in frames:
            base64_image = encode_image_to_base64(frame_path)
            if base64_image:
                image_data.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image
                    }
                })
        
        if not image_data:
            raise HTTPException(status_code=500, detail="Failed to process video frames for analysis")
        
        # Create prompt for Claude
        duration = int(metadata.get('duration', 30))
        
        claude_prompt = f"""You are a professional music producer and songwriter. Analyze these video frames and create ACTUAL LYRICS for a real song, not a description.

VIDEO CONTEXT:
- Duration: {duration} seconds
- Resolution: {metadata.get('width', 1920)}x{metadata.get('height', 1080)}

CRITICAL: Generate REAL LYRICS that sound like an actual song, not generic AI music.

OUTPUT FORMAT - WRITE ACTUAL LYRICS:
[Verse 1]
[Your lyrics here]

[Chorus]
[Your lyrics here]

[Verse 2]
[Your lyrics here]

[Chorus]
[Your lyrics here]

[Bridge]
[Your lyrics here]

[Outro]
[Your lyrics here]

LYRICS REQUIREMENTS:
- Write actual song lyrics with verses, chorus, bridge
- Make them emotional and relatable
- Use vivid imagery and metaphors
- Include repetition for catchiness
- Match the mood and energy of the video
- Sound like a real artist wrote them
- Avoid generic or cliché phrases

STYLE INSPIRATION (choose one that fits the video):
- Indie rock like Tame Impala, Radiohead, or Bon Iver
- Electronic pop like Daft Punk, ODESZA, or Flume
- Acoustic folk like Bon Iver, Fleet Foxes, or Iron & Wine
- Alternative rock like Radiohead, Arcade Fire, or The National
- Dream pop like Beach House, Cocteau Twins, or Mazzy Star

EXAMPLES OF GOOD LYRICS:
[Verse 1]
Driving through the neon lights
City dreams in my mind tonight
Every corner holds a story
Every moment feels like glory

[Chorus]
We're alive, we're breathing
In this moment we're believing
Nothing can stop us now
We're alive, we're breathing

Write lyrics that match the video's mood and energy. Make them sound like a real song."""

        # Call Claude API
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": claude_prompt}
                    ] + image_data
                }
            ]
        )
        
        prompt = message.content[0].text.strip()
        
        # Clean up frame files
        for frame_path in frames:
            try:
                os.remove(frame_path)
            except:
                pass
        
        return prompt
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude analysis failed: {str(e)}")

async def generate_music_prompt(metadata: dict, video_path: str) -> str:
    """Generate music prompt using Claude API"""
    if not CLAUDE_API_KEY or CLAUDE_API_KEY == "your_claude_api_key_here":
        raise HTTPException(status_code=500, detail="Claude API key not configured")
    
    return await analyze_video_with_claude(video_path, metadata)

def extract_tags_from_prompt(prompt: str) -> str:
    """Extract relevant tags from the lyrics prompt for Suno API"""
    prompt_lower = prompt.lower()
    
    # Genre detection based on lyrics content and style references
    genres = []
    if any(word in prompt_lower for word in ['electronic', 'edm', 'dance', 'synth', 'daft punk', 'odesza', 'flume', 'neon', 'digital']):
        genres.append('electronic pop')
    if any(word in prompt_lower for word in ['indie', 'indie rock', 'alternative', 'tame impala', 'bon iver', 'radiohead', 'arcade fire']):
        genres.append('indie rock')
    if any(word in prompt_lower for word in ['rock', 'guitar', 'electric', 'alternative rock', 'the national']):
        genres.append('alternative rock')
    if any(word in prompt_lower for word in ['pop', 'catchy', 'melodic', 'mainstream', 'uplifting']):
        genres.append('pop')
    if any(word in prompt_lower for word in ['ambient', 'chill', 'lo-fi', 'relaxing', 'atmospheric', 'dreamy', 'beach house']):
        genres.append('dream pop')
    if any(word in prompt_lower for word in ['hip-hop', 'rap', 'urban', 'trap', 'street']):
        genres.append('hip-hop')
    if any(word in prompt_lower for word in ['folk', 'acoustic', 'singer-songwriter', 'intimate', 'fleet foxes', 'iron & wine']):
        genres.append('indie folk')
    if any(word in prompt_lower for word in ['jazz', 'blues', 'soul', 'funk', 'smooth']):
        genres.append('jazz')
    if any(word in prompt_lower for word in ['orchestral', 'cinematic', 'orchestra', 'strings', 'classical', 'epic']):
        genres.append('cinematic')
    
    # Mood detection from lyrics
    moods = []
    if any(word in prompt_lower for word in ['energetic', 'upbeat', 'fast', 'driving', 'pumping', 'alive', 'breathing', 'glory']):
        moods.append('energetic')
    if any(word in prompt_lower for word in ['calm', 'peaceful', 'serene', 'gentle', 'dreamy', 'soft', 'quiet']):
        moods.append('calm')
    if any(word in prompt_lower for word in ['dramatic', 'epic', 'cinematic', 'powerful', 'intense', 'stories']):
        moods.append('dramatic')
    if any(word in prompt_lower for word in ['happy', 'joyful', 'uplifting', 'positive', 'bright', 'believing']):
        moods.append('uplifting')
    if any(word in prompt_lower for word in ['melancholy', 'sad', 'emotional', 'nostalgic', 'introspective', 'lonely']):
        moods.append('melancholy')
    if any(word in prompt_lower for word in ['raw', 'intimate', 'personal', 'vulnerable', 'real']):
        moods.append('intimate')
    
    # Production style detection
    production = []
    if any(word in prompt_lower for word in ['vintage', 'retro', 'analog', 'tape', 'vinyl', 'warm']):
        production.append('vintage')
    if any(word in prompt_lower for word in ['modern', 'digital', 'crisp', 'clean', 'polished', 'neon']):
        production.append('modern')
    if any(word in prompt_lower for word in ['reverb', 'atmospheric', 'spacey', 'ethereal', 'dreamy']):
        production.append('atmospheric')
    if any(word in prompt_lower for word in ['raw', 'live', 'organic', 'natural', 'acoustic']):
        production.append('organic')
    
    # Combine tags with better defaults
    all_tags = genres + moods + production
    if not all_tags:
        all_tags = ['indie rock', 'emotional', 'authentic']
    
    return ', '.join(all_tags)

async def call_suno_api(prompt: str, duration: int = 30) -> str:
    """Call Suno API to generate music using official HackMIT API format"""
    headers = {
        "Authorization": f"Bearer {SUNO_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Extract tags from prompt for better matching
    tags = extract_tags_from_prompt(prompt)
    
    # Use official HackMIT API format - prompt should contain actual lyrics
    data = {
        "prompt": prompt,  # This should be the actual lyrics we generated
        "tags": tags,      # Musical style tags
        "make_instrumental": False  # Allow vocals since we have lyrics
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
            
            # Official API returns a single clip object
            if isinstance(result, dict) and "id" in result:
                return result["id"]
            else:
                raise Exception("Unexpected response format from Suno API")
                
        except Exception as e:
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
                
                if status == "complete" or status == "streaming":
                    return clip.get("audio_url")
            return None
        except Exception as e:
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
            raise

async def trim_audio(input_path: str, output_path: str, duration: float):
    """Trim audio to match video duration asynchronously"""
    cmd = [
        'ffmpeg', '-y', '-v', 'quiet',
        '-i', input_path,
        '-t', str(duration),
        '-acodec', 'mp3',
        output_path
    ]
    
    returncode, stdout, stderr = await run_ffmpeg_async(cmd, timeout=60)
    
    if returncode != 0:
        # If trimming fails, just copy the original file
        import shutil
        shutil.copy2(input_path, output_path)

async def convert_flv_to_mp4(flv_path: str, mp4_path: str):
    """Convert FLV to MP4 using ffmpeg asynchronously"""
    cmd = [
        'ffmpeg', '-y', '-v', 'quiet',
        '-i', flv_path,
        '-vcodec', 'libx264',
        '-acodec', 'aac',
        mp4_path
    ]
    
    returncode, stdout, stderr = await run_ffmpeg_async(cmd, timeout=300)
    
    if returncode != 0:
        raise Exception(f"FFmpeg conversion failed: {stderr}")

async def merge_video_audio(video_path: str, audio_path: str, output_path: str):
    """Replace video audio with Suno music only"""
    print(f"🔧 MERGING: Replacing video audio with Suno music only")
    print(f"   Video: {video_path}")
    print(f"   Audio: {audio_path}")
    print(f"   Output: {output_path}")
    
    cmd = [
        'ffmpeg', '-y', '-v', 'quiet',
        '-i', video_path,
        '-i', audio_path,
        '-map', '0:v',  # Use video from first input
        '-map', '1:a',  # Use audio from second input (Suno music)
        '-vcodec', 'copy',
        '-acodec', 'aac',
        output_path
    ]
    
    print(f"   Running FFmpeg command...")
    returncode, stdout, stderr = await run_ffmpeg_async(cmd, timeout=300)
    
    if returncode != 0:
        print(f"❌ FFmpeg merge failed: {stderr}")
        raise Exception(f"FFmpeg merge failed: {stderr}")
    else:
        print(f"✅ FFmpeg merge completed successfully")

async def cleanup_temp_files(task_id: str):
    """Clean up temporary files"""
    try:
        task = tasks.get(task_id)
        if not task:
            return
            
        # Remove video files
        if task.video_path and os.path.exists(task.video_path):
            os.remove(task.video_path)
        
        # Keep music files for debugging - don't clean up
        print(f"🎵 MUSIC SAVED: {task.music_path}")
        if '_trimmed' in task.music_path:
            original_music_path = task.music_path.replace('_trimmed', '')
            print(f"🎵 ORIGINAL MUSIC SAVED: {original_music_path}")
            
    except Exception as e:
        print(f"Cleanup error: {e}")

async def process_video_task(task_id: str):
    """Background task to process video with concurrency control"""
    global active_tasks
    
    # Acquire semaphore to limit concurrent tasks
    async with task_semaphore:
        active_tasks += 1
        print(f"Starting task {task_id} (active: {active_tasks})")
        
        try:
            task = tasks.get(task_id)
            if not task:
                return
            
            await _process_video_task_internal(task_id, task)
        finally:
            active_tasks -= 1
            print(f"Completed task {task_id} (active: {active_tasks})")

async def _process_video_task_internal(task_id: str, task: Task):
    """Internal video processing logic"""
    try:
        # Step 1: Convert FLV to MP4
        print(f"🔄 CONVERTING VIDEO: FLV to MP4...")
        task.status = TaskStatus.CONVERTING
        task.progress = 20
        task.updated_at = datetime.now()
        
        flv_path = task.video_path
        mp4_path = str(TEMP_DIR / "videos" / f"{task_id}.mp4")
        print(f"   Input: {flv_path}")
        print(f"   Output: {mp4_path}")
        
        await convert_flv_to_mp4(flv_path, mp4_path)
        task.video_path = mp4_path
        print(f"✅ Video conversion complete: {mp4_path}")
        
        # Step 2: Analyze video and generate music
        task.status = TaskStatus.GENERATING
        task.progress = 40
        task.updated_at = datetime.now()
        
        print(f"📊 ANALYZING VIDEO: Getting metadata...")
        metadata = await get_video_metadata(mp4_path)
        print(f"✅ Video metadata: {metadata}")
        
        print(f"🤖 GENERATING PROMPT: Calling Claude API...")
        prompt = await generate_music_prompt(metadata, mp4_path)
        print(f"✅ Generated prompt: {prompt[:100]}...")
        
        # Call Suno API
        print(f"🎵 CALLING SUNO API: Generating music...")
        suno_task_id = await call_suno_api(prompt, int(metadata.get('duration', 30)))
        print(f"✅ Suno API called, task ID: {suno_task_id}")
        
        # Poll for completion
        print(f"⏳ POLLING SUNO: Waiting for music generation...")
        audio_url = None
        max_attempts = 30  # 5 minutes max
        for attempt in range(max_attempts):
            await asyncio.sleep(10)  # Wait 10 seconds between polls
            print(f"   Polling attempt {attempt + 1}/{max_attempts}...")
            audio_url = await poll_suno_status(suno_task_id)
            if audio_url:
                print(f"✅ Music generation complete! Audio URL: {audio_url}")
                break
            # More realistic progress: 40% + (attempts * 1.5%) up to 85%
            task.progress = min(40 + (attempt * 1.5), 85)
            task.updated_at = datetime.now()
        
        if not audio_url:
            raise Exception(f"Music generation timeout after {max_attempts * 10} seconds")
        
        # Download music
        print(f"📥 DOWNLOADING MUSIC: Getting audio from Suno...")
        music_path = str(TEMP_DIR / "music" / f"{task_id}.mp3")
        await download_music(audio_url, music_path)
        print(f"✅ Music downloaded: {music_path}")
        
        # Trim music to match video duration
        print(f"✂️  TRIMMING MUSIC: Matching video duration...")
        video_duration = metadata.get('duration', 30)
        trimmed_music_path = str(TEMP_DIR / "music" / f"{task_id}_trimmed.mp3")
        await trim_audio(music_path, trimmed_music_path, video_duration)
        task.music_path = trimmed_music_path
        print(f"✅ Music trimmed: {trimmed_music_path}")
        
        # Step 3: Merge video and audio
        task.status = TaskStatus.MERGING
        task.progress = 90
        task.updated_at = datetime.now()
        
        output_path = str(TEMP_DIR / "output" / f"{task_id}.mp4")
        await merge_video_audio(mp4_path, trimmed_music_path, output_path)
        task.output_path = output_path
        
        # Step 4: Complete
        task.status = TaskStatus.DONE
        task.progress = 100
        task.updated_at = datetime.now()
        
    except Exception as e:
        task.status = TaskStatus.ERROR
        task.error_message = str(e)
        task.updated_at = datetime.now()

# API Endpoints
@app.post("/upload", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload FLV video from RTMP server"""
    
    # Validate file
    if file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    if not file.filename or not file.filename.lower().endswith('.flv'):
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
        "total_tasks": len(tasks),
        "active_tasks": active_tasks,
        "max_concurrent": MAX_CONCURRENT_TASKS
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
