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
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from anthropic import Anthropic
from PIL import Image

# Load environment variables
load_dotenv()

# Configuration
SUNO_API_KEY = os.getenv("SUNO_API_KEY", "your_suno_api_key_here")
SUNO_BASE_URL = os.getenv("SUNO_BASE_URL", "https://studio-api.prod.suno.com/api/v2/external/hackmit")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "your_claude_api_key_here")
TEMP_DIR = Path(os.getenv("TEMP_DIR", "temp"))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB for photos

# Ensure temp directories exist
TEMP_DIR.mkdir(exist_ok=True)
(TEMP_DIR / "photos").mkdir(exist_ok=True)
(TEMP_DIR / "music").mkdir(exist_ok=True)
(TEMP_DIR / "output").mkdir(exist_ok=True)

# FastAPI app
app = FastAPI(title="Photo to Video Processor", version="1.0.0")

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"🌐 REQUEST: {request.method} {request.url}")
    print(f"   Headers: {dict(request.headers)}")
    print(f"   Client: {request.client}")
    
    response = await call_next(request)
    
    print(f"📤 RESPONSE: {response.status_code}")
    return response

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
    photo_path: str
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

def encode_image_to_base64(image_path: str) -> str:
    """Convert image to base64 for Claude API"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Image encoding error: {e}")
        return ""

async def analyze_photo_with_claude(photo_path: str) -> str:
    """Use Claude API to analyze photo and generate music prompt"""
    try:
        # Initialize Claude client
        client = Anthropic(api_key=CLAUDE_API_KEY)
        
        # Encode photo for Claude
        base64_image = encode_image_to_base64(photo_path)
        if not base64_image:
            raise HTTPException(status_code=500, detail="Failed to process photo for analysis")
        
        # Create prompt for Claude
        claude_prompt = """You are a professional music producer and songwriter. Analyze this photo and create ACTUAL LYRICS for a real song, not a description.

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
- Match the mood and energy of the photo
- Sound like a real artist wrote them
- Avoid generic or cliché phrases

STYLE INSPIRATION (choose one that fits the photo):
- Indie rock like Tame Impala, Radiohead, or Bon Iver
- Electronic pop like Daft Punk, ODESZA, or Flume
- Acoustic folk like Bon Iver, Fleet Foxes, or Iron & Wine
- Alternative rock like Radiohead, Arcade Fire, or The National
- Dream pop like Beach House, Cocteau Twins, or Mazzy Star

Write lyrics that match the photo's mood and energy. Make them sound like a real song."""

        # Call Claude API
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": claude_prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        }
                    ]
                }
            ]
        )
        
        prompt = message.content[0].text.strip()
        return prompt
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claude analysis failed: {str(e)}")

async def generate_music_prompt(photo_path: str) -> str:
    """Generate music prompt using Claude API"""
    if not CLAUDE_API_KEY or CLAUDE_API_KEY == "your_claude_api_key_here":
        raise HTTPException(status_code=500, detail="Claude API key not configured")
    
    return await analyze_photo_with_claude(photo_path)

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

async def call_suno_api(prompt: str, duration: int = 10) -> str:
    """Call Suno API to generate music using official HackMIT API format"""
    print(f"🎵 SUNO API CALL: Starting music generation")
    print(f"   API Key present: {bool(SUNO_API_KEY and SUNO_API_KEY != 'your_suno_api_key_here')}")
    print(f"   Base URL: {SUNO_BASE_URL}")
    print(f"   Prompt length: {len(prompt)} characters")
    
    headers = {
        "Authorization": f"Bearer {SUNO_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Extract tags from prompt for better matching
    tags = extract_tags_from_prompt(prompt)
    print(f"   Extracted tags: {tags}")
    
    # Use official HackMIT API format - prompt should contain actual lyrics
    data = {
        "prompt": prompt,  # This should be the actual lyrics we generated
        "tags": tags,      # Musical style tags
        "make_instrumental": False  # Allow vocals since we have lyrics
    }
    
    print(f"   Request data: {data}")
    
    async with httpx.AsyncClient() as client:
        try:
            print(f"   Making POST request to: {SUNO_BASE_URL}/generate")
            response = await client.post(
                f"{SUNO_BASE_URL}/generate",
                headers=headers,
                json=data,
                timeout=30.0
            )
            
            print(f"   Response status: {response.status_code}")
            print(f"   Response headers: {dict(response.headers)}")
            
            response.raise_for_status()
            result = response.json()
            
            print(f"   Response body: {result}")
            
            # Official API returns a single clip object
            if isinstance(result, dict) and "id" in result:
                print(f"✅ Suno API success: Task ID {result['id']}")
                return result["id"]
            else:
                error_msg = f"Unexpected response format from Suno API: {result}"
                print(f"❌ {error_msg}")
                raise Exception(error_msg)
                
        except httpx.HTTPStatusError as e:
            error_msg = f"Suno API HTTP error {e.response.status_code}: {e.response.text}"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        except httpx.TimeoutException as e:
            error_msg = f"Suno API timeout: {str(e)}"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        except httpx.RequestError as e:
            error_msg = f"Suno API request error: {str(e)}"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)
        except Exception as e:
            error_msg = f"Suno API unexpected error: {str(e)}"
            print(f"❌ {error_msg}")
            raise HTTPException(status_code=500, detail=error_msg)

async def poll_suno_status(suno_task_id: str) -> Optional[str]:
    """Poll Suno API for music generation status"""
    print(f"⏳ POLLING SUNO: Checking status for task {suno_task_id}")
    
    headers = {
        "Authorization": f"Bearer {SUNO_API_KEY}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            print(f"   Making GET request to: {SUNO_BASE_URL}/clips?ids={suno_task_id}")
            response = await client.get(
                f"{SUNO_BASE_URL}/clips?ids={suno_task_id}",
                headers=headers,
                timeout=30.0
            )
            
            print(f"   Response status: {response.status_code}")
            response.raise_for_status()
            result = response.json()
            
            print(f"   Response body: {result}")
            
            # The response is an array of clips
            if isinstance(result, list) and len(result) > 0:
                clip = result[0]
                status = clip.get("status")
                print(f"   Clip status: {status}")
                
                if status == "complete" or status == "streaming":
                    audio_url = clip.get("audio_url")
                    print(f"✅ Music ready! Audio URL: {audio_url}")
                    return audio_url
                elif status == "failed":
                    error_msg = f"Music generation failed: {clip.get('error', 'Unknown error')}"
                    print(f"❌ {error_msg}")
                    raise Exception(error_msg)
                else:
                    print(f"   Status: {status} - continuing to poll...")
                    return None
            else:
                print(f"   No clips found in response")
                return None
                
        except httpx.HTTPStatusError as e:
            error_msg = f"Suno polling HTTP error {e.response.status_code}: {e.response.text}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
        except httpx.TimeoutException as e:
            error_msg = f"Suno polling timeout: {str(e)}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
        except httpx.RequestError as e:
            error_msg = f"Suno polling request error: {str(e)}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)
        except Exception as e:
            error_msg = f"Suno polling unexpected error: {str(e)}"
            print(f"❌ {error_msg}")
            raise Exception(error_msg)

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

async def create_video_from_photo(photo_path: str, output_path: str, duration: int = 10):
    """Create a static video from photo using FFmpeg"""
    cmd = [
        'ffmpeg', '-y', '-v', 'quiet',
        '-loop', '1',
        '-i', photo_path,
        '-c:v', 'libx264',
        '-t', str(duration),
        '-pix_fmt', 'yuv420p',
        '-vf', 'scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2',
        output_path
    ]
    
    returncode, stdout, stderr = await run_ffmpeg_async(cmd, timeout=120)
    
    if returncode != 0:
        raise Exception(f"FFmpeg video creation failed: {stderr}")

async def merge_video_audio(video_path: str, audio_path: str, output_path: str):
    """Replace video audio with Suno music only"""
    print(f"🔧 MERGING: Adding music to video")
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
            
        # Remove photo files
        if task.photo_path and os.path.exists(task.photo_path):
            os.remove(task.photo_path)
        
        # Keep music files for debugging - don't clean up
        print(f"🎵 MUSIC SAVED: {task.music_path}")
        if '_trimmed' in task.music_path:
            original_music_path = task.music_path.replace('_trimmed', '')
            print(f"🎵 ORIGINAL MUSIC SAVED: {original_music_path}")
            
    except Exception as e:
        print(f"Cleanup error: {e}")

async def process_photo_task(task_id: str):
    """Background task to process photo with concurrency control"""
    global active_tasks
    
    # Acquire semaphore to limit concurrent tasks
    async with task_semaphore:
        active_tasks += 1
        print(f"Starting task {task_id} (active: {active_tasks})")
        
        try:
            task = tasks.get(task_id)
            if not task:
                return
            
            await _process_photo_task_internal(task_id, task)
        finally:
            active_tasks -= 1
            print(f"Completed task {task_id} (active: {active_tasks})")

async def _process_photo_task_internal(task_id: str, task: Task):
    """Internal photo processing logic"""
    try:
        # Step 1: Create video from photo
        print(f"🔄 CONVERTING PHOTO: Creating video from photo...")
        task.status = TaskStatus.CONVERTING
        task.progress = 20
        task.updated_at = datetime.now()
        
        photo_path = task.photo_path
        video_path = str(TEMP_DIR / "output" / f"{task_id}_temp.mp4")
        print(f"   Input: {photo_path}")
        print(f"   Output: {video_path}")
        
        await create_video_from_photo(photo_path, video_path, duration=10)
        print(f"✅ Video creation complete: {video_path}")
        
        # Step 2: Analyze photo and generate music
        task.status = TaskStatus.GENERATING
        task.progress = 40
        task.updated_at = datetime.now()
        
        print(f"🤖 GENERATING PROMPT: Calling Claude API...")
        prompt = await generate_music_prompt(photo_path)
        print(f"✅ Generated prompt: {prompt[:100]}...")
        
        # Call Suno API
        print(f"🎵 CALLING SUNO API: Generating music...")
        suno_task_id = await call_suno_api(prompt, duration=10)
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
        trimmed_music_path = str(TEMP_DIR / "music" / f"{task_id}_trimmed.mp3")
        await trim_audio(music_path, trimmed_music_path, 10.0)
        task.music_path = trimmed_music_path
        print(f"✅ Music trimmed: {trimmed_music_path}")
        
        # Step 3: Merge video and audio
        task.status = TaskStatus.MERGING
        task.progress = 90
        task.updated_at = datetime.now()
        
        output_path = str(TEMP_DIR / "output" / f"{task_id}.mp4")
        await merge_video_audio(video_path, trimmed_music_path, output_path)
        task.output_path = output_path
        
        # Clean up temp video
        if os.path.exists(video_path):
            os.remove(video_path)
        
        # Step 4: Complete
        task.status = TaskStatus.DONE
        task.progress = 100
        task.updated_at = datetime.now()
        
    except Exception as e:
        import traceback
        error_details = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(f"❌ TASK FAILED: {error_details}")
        
        task.status = TaskStatus.ERROR
        task.error_message = error_details
        task.updated_at = datetime.now()

# API Endpoints
@app.get("/test-upload")
async def test_upload():
    """Test endpoint to verify routing"""
    return {"message": "Upload endpoint routing is working", "status": "ok"}

@app.post("/upload-photo", response_model=UploadResponse)
async def upload_photo(file: UploadFile = File(...)):
    """Upload photo from glasses"""
    
    print(f"📸 UPLOAD REQUEST: Received photo upload request")
    print(f"   Filename: {file.filename}")
    print(f"   Content type: {file.content_type}")
    print(f"   File size: {file.size}")
    
    # Validate file
    if file.size > MAX_FILE_SIZE:
        print(f"❌ File too large: {file.size} > {MAX_FILE_SIZE}")
        raise HTTPException(status_code=413, detail="File too large")
    
    if not file.filename or not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=400, detail="Only JPG/PNG files are supported")
    
    # Generate task ID and save file
    task_id = generate_task_id()
    photo_path = str(TEMP_DIR / "photos" / f"{task_id}.jpg")
    
    try:
        with open(photo_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")
    
    # Create task
    task = Task(
        task_id=task_id,
        status=TaskStatus.UPLOADED,
        progress=0,
        photo_path=photo_path,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    tasks[task_id] = task
    
    # Start processing immediately
    asyncio.create_task(process_photo_task(task_id))
    
    return UploadResponse(
        task_id=task_id,
        status="uploaded",
        message="Photo uploaded and processing started"
    )

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
    """Download final processed video"""
    
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
        filename=f"processed_video_{task_id}.mp4",
        media_type="video/mp4"
    )

@app.get("/library")
async def get_video_library():
    """Get library of all processed videos"""
    output_dir = TEMP_DIR / "output"
    
    if not output_dir.exists():
        return {"videos": []}
    
    videos = []
    
    # Get all MP4 files in output directory
    for video_file in output_dir.glob("*.mp4"):
        try:
            # Get file stats
            stat = video_file.stat()
            file_size = stat.st_size
            created_time = datetime.fromtimestamp(stat.st_ctime)
            modified_time = datetime.fromtimestamp(stat.st_mtime)
            
            # Extract task_id from filename (filename is task_id.mp4)
            task_id = video_file.stem
            
            # Get task info if available
            task_info = tasks.get(task_id, {})
            
            video_info = {
                "task_id": task_id,
                "filename": video_file.name,
                "file_size": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "created_at": created_time.isoformat(),
                "modified_at": modified_time.isoformat(),
                "download_url": f"/download/{task_id}",
                "status": task_info.get("status", "unknown") if task_info else "unknown",
                "progress": task_info.get("progress", 0) if task_info else 0
            }
            
            videos.append(video_info)
            
        except Exception as e:
            print(f"Error processing video file {video_file}: {e}")
            continue
    
    # Sort by creation time (newest first)
    videos.sort(key=lambda x: x["created_at"], reverse=True)
    
    return {
        "videos": videos,
        "total_count": len(videos),
        "total_size_mb": sum(v["file_size_mb"] for v in videos)
    }

@app.get("/library/{task_id}")
async def get_video_info(task_id: str):
    """Get detailed info about a specific video"""
    output_file = TEMP_DIR / "output" / f"{task_id}.mp4"
    
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    try:
        # Get file stats
        stat = output_file.stat()
        file_size = stat.st_size
        created_time = datetime.fromtimestamp(stat.st_ctime)
        modified_time = datetime.fromtimestamp(stat.st_mtime)
        
        # Get task info if available
        task_info = tasks.get(task_id, {})
        
        video_info = {
            "task_id": task_id,
            "filename": output_file.name,
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "created_at": created_time.isoformat(),
            "modified_at": modified_time.isoformat(),
            "download_url": f"/download/{task_id}",
            "status": task_info.get("status", "unknown") if task_info else "unknown",
            "progress": task_info.get("progress", 0) if task_info else 0,
            "photo_path": task_info.get("photo_path") if task_info else None,
            "music_path": task_info.get("music_path") if task_info else None
        }
        
        return video_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting video info: {str(e)}")

@app.get("/debug/task/{task_id}")
async def debug_task(task_id: str):
    """Debug endpoint to get detailed task information"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    # Check if files exist
    photo_exists = os.path.exists(task.photo_path) if task.photo_path else False
    music_exists = os.path.exists(task.music_path) if task.music_path else False
    output_exists = os.path.exists(task.output_path) if task.output_path else False
    
    return {
        "task_id": task_id,
        "status": task.status.value,
        "progress": task.progress,
        "error_message": task.error_message,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "files": {
            "photo_path": task.photo_path,
            "photo_exists": photo_exists,
            "music_path": task.music_path,
            "music_exists": music_exists,
            "output_path": task.output_path,
            "output_exists": output_exists
        },
        "environment": {
            "suno_api_key_set": bool(SUNO_API_KEY and SUNO_API_KEY != "your_suno_api_key_here"),
            "claude_api_key_set": bool(CLAUDE_API_KEY and CLAUDE_API_KEY != "your_claude_api_key_here"),
            "suno_base_url": SUNO_BASE_URL
        }
    }

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Photo to Video Processor",
        "status": "running",
        "total_tasks": len(tasks),
        "active_tasks": active_tasks,
        "max_concurrent": MAX_CONCURRENT_TASKS
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)