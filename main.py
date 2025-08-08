import shutil
import os
import uuid
from typing import List, Dict, Any
import time
import sys
import cv2
sys.path.append("/root/utils")

from utils.video_utils import extract_video_frames
from utils.nsfw_utils import scan_images_for_nsfw
from utils.helpers import validate_input, download_file, wait_for_done

# ================= Constants =================
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MAX_VIDEO_SIZE_MB = 100 * 1024 * 1024  # 100MB
MAX_IMAGE_SIZE_MB = 10  # 10MB
MAX_VIDEO_DURATION = 5 * 60  # 5 minutes in seconds
BASE_DIR = "/tmp/data"

# This function should return a map of jobs e.g for images {job_id:image_path} e.g for videos {job_id: {frames:[frame_path1, frame_path2, ...], metadata: {width, height, duration, fps, size_mb}}}
# ================= Preprocessing Function (CPU-only) =================
def preprocess_media(media_type: str, jobs: List[Dict[str, str]], round_id: str) -> List[Dict[str, Any]]:
    """
    Jobs look like this:{"type": "videos", "jobs": [{"job_id": "1", "url": "http://example.com/video.mp4"}]}
    Downloads media files, extracts frames for videos, and stores them in shared volume.
    Returns a map of jobs.
    Preprocessed video jobs will look like this:
    [{ "job_id": "1",
        "frames_paths": ["/data/{round_id}/videos/{job_id}/frames/frame_0001.jpg", ...],
        "metadata": { "width": 1920, "height": 1080, "duration": 120, "fps": 30, "size_mb": 100 }
        }]
    Preprocessed image jobs will look like this:
    [{ "job_id": "1", "image_path": "/data/{round_id}/images/{job_id}.jpg" }]
    """
    try: 
        validate_input({"type": media_type, "jobs": jobs})
        print(f"[PREPROCESS] Input validated. Media type: {media_type}, Number of jobs: {len(jobs)}")

        preprocessed_jobs = []

        if media_type == "images":
            # Preprocess images
            for job in jobs:
                job_id = job["job_id"]
                url = job["url"]
                work_dir = os.path.join(BASE_DIR, round_id, "images")
                os.makedirs(work_dir, exist_ok=True)
                print(f"[PREPROCESS] Created work_dir: {work_dir}")

                path = download_file(url, work_dir, filename=job_id, max_bytes=MAX_IMAGE_SIZE_MB)
                if not path or os.path.splitext(path)[-1].lower() not in ALLOWED_IMAGE_EXTENSIONS:
                    print(f"[PREPROCESS] Skipping invalid or unsupported image: {url}")
                    continue

                print(f"[PREPROCESS] Image downloaded to: {path}")

                preprocessed_jobs.append({"job_id": job_id, "image_path": path})
        elif media_type == "videos":
            for job in jobs:
                job_id = job["job_id"]
                url = job["url"]
                work_dir = os.path.join(BASE_DIR, round_id, "videos", job_id)
                os.makedirs(work_dir, exist_ok=True)
                print(f"[PREPROCESS] Created work_dir: {work_dir}")

                video_path = download_file(url, work_dir, max_bytes=MAX_VIDEO_SIZE_MB)
                if not video_path or os.path.splitext(video_path)[-1].lower() not in ALLOWED_VIDEO_EXTENSIONS:
                    print(f"[PREPROCESS] Skipping invalid or unsupported video: {url}")
                    continue
                
                print(f"[PREPROCESS] Video downloaded to: {video_path}")
                
                preprocessed_jobs.append({"job_id": job_id, "frames_paths": [], "metadata": {}})

                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    continue
                print(f"[PREPROCESS] Opened video: {video_path}")

                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                if fps <= 0 or frame_count <= 0:
                    continue
                
                duration = frame_count / fps
                if duration > MAX_VIDEO_DURATION:
                    continue
                
                print(f"[PREPROCESS] Extracting frames from: {video_path} into: {work_dir}")
                frames = extract_video_frames(video_path, work_dir)
                # Wait for done.json to check if all the i/o operation on the files are done.
                done_path = os.path.join(work_dir, "done.json")
                done_metadata = wait_for_done(done_path)
                print(f"[PREPROCESS] Extracted {len(frames)} frames.")

                video_metadata = {
                    "width": width,
                    "height": height,
                    "duration": round(duration, 2),
                    "fps": round(fps, 2),
                    "size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2),
                    "original_filename": os.path.basename(video_path)
                }
                preprocessed_jobs[-1]["frames_paths"] = frames
                preprocessed_jobs[-1]["metadata"] = video_metadata

    except Exception as e:
        print(f"Preprocessing failed: {e}")
        raise

    return preprocessed_jobs

# ================= Scanning Function (GPU) =================
def scan_frames(media_type: str, frame_paths: List[str]) -> List[Dict]:
    """
    Processes frames through NSFW detection model.
    Returns list of results with NSFW scores and metadata.
    """
    if not frame_paths:
        return []
    
    for attempt in range(5):
        missing = [p for p in frame_paths if not os.path.exists(p)]
        if not missing:
            break
        print(f"[WARN] Waiting for {len(missing)} frames to appear...")
        time.sleep(0.5)
    else:
        raise RuntimeError(f"[ERROR] Final frames missing: {missing}")

    try:
        # Batch process all frames
        print(f"[SCAN] Running NSFW scan on frames: {frame_paths[0:3]}...")
        scan_results = scan_images_for_nsfw(frame_paths)
        print(f"[SCAN] Scan complete. Got {len(scan_results)} results.")
        
        results = []
        for frame_path, (_, is_nsfw, nsfw_score, full_probs) in zip(frame_paths, scan_results):
            # Get metadata for the frame
            if media_type == "images":
                img = cv2.imread(frame_path)
                if img is None:
                    raise ValueError(f"Could not read image: {frame_path}")

                height, width = img.shape[:2]
                size_mb = os.path.getsize(frame_path) / (1024 * 1024)
                results.append({
                    "filename": os.path.basename(frame_path),
                    "relative_path": os.path.relpath(frame_path, BASE_DIR),
                    "is_nsfw": is_nsfw,
                    "nsfw_score": nsfw_score,
                    "full_probs": full_probs,
                    "width": width,
                    "height": height,
                    "size_mb": round(size_mb, 2),
                    "error": None
                })
            else:  # video frames
                results.append({
                    "filename": os.path.basename(frame_path),
                    "relative_path": os.path.relpath(frame_path, BASE_DIR),
                    "is_nsfw": is_nsfw,
                    "nsfw_score": nsfw_score,
                    "full_probs": full_probs,
                    "error": None
                })
        
        return results
    
    except Exception as e:
        print(f"Scanning failed: {e}")
        print(f"[ERROR] Failed during scanning. Paths: {frame_paths[0:3]}...")
        return [{
            "filename": os.path.basename(p),
            "relative_path": os.path.relpath(p, "/data"),  # ensure this is always present
            "is_nsfw": False,
            "nsfw_score": 0.0,
            "full_probs": {},
            "error": str(e)
        } for p in frame_paths]


# ================= Entrypoint Function (CPU-only) =================
def analyze_media(request: Dict) -> Dict:
    """
    Public-facing endpoint that orchestrates the preprocessing and scanning.
    """
    round_id = str(uuid.uuid4())  # Unique ID for this analysis round
    validate_input(request)
    media_type = request["type"]
    jobs = request["jobs"]

    # Step 1: Preprocess media (download, validate, extract)
    preprocessed_jobs = preprocess_media(media_type, jobs, round_id)
    print(f"[PREPROCESS] Preprocessed {len(preprocessed_jobs)} jobs for {media_type}.")

    scan_results = []

    if media_type == "images":
        # Batch scan all images together
        frame_paths = [job["image_path"] for job in preprocessed_jobs]
        print(f"[SCAN] Scanning {len(frame_paths)} images...")
        scan_results = scan_frames(media_type, frame_paths)

        # Augment with metadata
        scan_results = [
            {
                **result,
                "frame_count": 1,
                "engine_version": "scan-engine-v1"
            }
            for result in scan_results
        ]

    elif media_type == "videos":
        video_results = []

        for job in preprocessed_jobs:
            job_id = job["job_id"]
            meta = job.get("metadata", {})
            frames = job.get("frames_paths", [])

            if not frames:
                print(f"[SCAN] No frames for job {job_id}, skipping.")
                continue

            print(f"[SCAN] Scanning {len(frames)} frames for job {job_id}...")
            frame_results = scan_frames(media_type, frames)  # or await/collect depending on your executor

            if not frame_results:
                print(f"[SCAN] No results for job {job_id}")
                continue

            most_nsfw = max(frame_results, key=lambda x: x["nsfw_score"])
            avg_score = sum(f["nsfw_score"] for f in frame_results) / len(frame_results)

            video_results.append({
                "filename": meta.get("original_filename", job_id),
                "is_nsfw": any(f["is_nsfw"] for f in frame_results),
                "nsfw_score_max": most_nsfw["nsfw_score"],
                "nsfw_score_avg": round(avg_score, 4),
                "full_probs": most_nsfw["full_probs"],
                "frame_count": len(frame_results),
                "engine_version": "scan-engine-v1",
                "width": meta.get("width"),
                "height": meta.get("height"),
                "duration": meta.get("duration"),
                "fps": meta.get("fps"),
                "size_mb": meta.get("size_mb"),
                "error": None
            })

            print(f"[SCAN] Finished job {job_id}.")

            # 🔁 Optional: stream / yield / save intermediate result here
            # yield or send result to message queue, socket, or db

        scan_results = video_results
        return {"results": scan_results}
    
    shutil.rmtree(os.path.join(BASE_DIR, round_id), ignore_errors=True)
    print(f"[CLEANUP] Removed: {os.path.join(BASE_DIR, round_id)}")

## Notes:

# The file structure in the shared volume will be:
# Images:
# - /data/{round_id}/images/{job_id}.{ext}
# Videos:
# - /data/{round_id}/videos/{job_id}/frames/frame_0001.jpg
# - /data/{round_id}/videos/{job_id}/video.mp4


# Input format for analyze_media:
# {
#     "type": "videos",  # or "images"
#     "jobs": [
#         {"job_id": "1", "url": "http://example.com/video.mp4"},
#         {"job_id": "2", "url": "http://example.com/image.jpg"}
#     ]