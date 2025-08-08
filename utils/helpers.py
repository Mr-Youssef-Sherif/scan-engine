import os
import uuid
import requests
from typing import Optional, Dict
from urllib.parse import urlparse
import time
import json

# ================= Constants =================
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# ================= Utility Functions =================
def download_file(url: str, dest_folder: str, filename: Optional[str] = None, max_bytes: int = MAX_FILE_SIZE) -> Optional[str]:
    """Downloads a file with size limit checks."""
    try:
        if not url.startswith(("http://", "https://")):
            raise ValueError("Invalid URL format")
        
        parsed = urlparse(url)
        ext = os.path.splitext(parsed.path)[-1].lower()
        if ext not in ALLOWED_VIDEO_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS:
            ext = ".bin"
        
        if filename:
            local_filename = f"{filename}{ext}"
        else:
            local_filename = f"{uuid.uuid4()}{ext}"
        
        path = os.path.join(dest_folder, local_filename)
        
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            total_downloaded = 0
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        total_downloaded += len(chunk)
                        if total_downloaded > max_bytes:
                            f.close()
                            os.remove(path)
                            raise ValueError("File exceeds size limit")
                        f.write(chunk)
        return path
    except requests.exceptions.RequestException:
        return None#, "Failed to download file – network issue or inaccessible URL"
    except Exception as e:
        return None#, f"Unexpected download error: {str(e)}"

def validate_input(req: Dict):
    """Validates the input request structure."""
    if not isinstance(req, dict):
        raise ValueError("Input must be a dictionary with 'type' and 'jobs'")
    if "type" not in req or "jobs" not in req:
        raise ValueError("Missing 'type' or 'jobs' keys")
    if req["type"] not in {"videos", "images"}:
        raise ValueError("Invalid type: must be 'videos' or 'images'")
    if not isinstance(req["jobs"], list) or not (1 <= len(req["jobs"]) <= 40):
        raise ValueError("'jobs' must be a list of 1 to 40 jobs")
    if req["type"] == "videos" and len(req["jobs"]) > 10:
        raise ValueError("Too many video jobs: max 10 allowed")
    if req["type"] == "images" and len(req["jobs"]) > 40:
        raise ValueError("Too many image jobs: max 40 allowed")
    if not all("job_id" in job and "url" in job for job in req["jobs"]):
        raise ValueError("Each job must contain 'job_id' and 'url' keys")
    

def wait_for_done(output_dir, timeout=60):
    done_path = os.path.join(output_dir)
    start = time.time()
    while not os.path.exists(done_path):
        if time.time() - start > timeout:
            raise TimeoutError(f"Timed out waiting for {done_path}")
        print(f"Waiting for {done_path} to appear...")
        time.sleep(1)
    with open(done_path) as f:
        meta = json.load(f)
    print(f"✅ Found metadata: {meta}")
    return meta