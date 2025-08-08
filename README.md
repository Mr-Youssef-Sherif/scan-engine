
# NSFW Media Scanner

A robust, scalable media scanner designed to **detect NSFW (Not Safe For Work) content** in both images and videos. This tool is built for high-throughput, automated content moderation pipelines, enabling reliable identification of inappropriate content across various media types.

----------

## Purpose

In modern content platforms, automatically filtering NSFW material is critical to maintain safe and compliant environments. This scanner serves as an **automated preprocessing and detection pipeline** that:

-   Downloads media from remote URLs,
    
-   Validates and enforces size and duration constraints,
    
-   Extracts representative frames from videos using advanced frame sampling techniques,
    
-   Performs GPU-accelerated NSFW classification on images and video frames,
    
-   Returns detailed, per-media-item and per-frame NSFW scores and metadata,
    
-   Ensures efficient resource use via frame deduplication and batch processing,
    
-   Cleans up temporary data to avoid storage bloat.
    

This system is ideal for integration in **content moderation workflows**, **automated review systems**, or any application needing scalable NSFW detection with fine-grained insight into media content.

----------

## What It Does

1.  **Input Handling & Validation:**  
    Accepts a batch of media jobs defined by `job_id` and `URL`, specifying whether the batch contains images or videos.
    
2.  **Media Download & Preprocessing:**
    
    -   **Images:** Downloads and validates format and size.
        
    -   **Videos:** Downloads, verifies file size and duration limits (max 100MB and 5 minutes).
        
    -   Extracts frames using a multi-pronged strategy:
        
        -   Scene-change detection via FFmpeg filters,
            
        -   Uniform sampling at a configurable low FPS,
            
        -   Luma spike detection for abrupt brightness changes.
            
    -   Deduplicates extracted frames via MD5 hashing to remove redundant frames, improving scanning efficiency.
        
3.  **NSFW Scanning:**  
    Runs a GPU-accelerated NSFW detection model (via `scan_images_for_nsfw`) on images or extracted frames, returning:
    
    -   NSFW classification (`is_nsfw` boolean),
        
    -   NSFW confidence scores,
        
    -   Detailed class probability distributions,
        
    -   Metadata such as dimensions, file size, and processing errors if any.
        
4.  **Results Aggregation:**  
    For videos, aggregates per-frame results to provide max and average NSFW scores, plus key metadata (resolution, duration, fps).
    
5.  **Cleanup:**  
    Removes temporary media files and extracted frames post-processing to maintain a clean workspace.
    

----------

## Technical Details

-   **Supported Formats:**
    
    -   Videos: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`
        
    -   Images: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
        
-   **Limits:**
    
    -   Max video size: 100MB
        
    -   Max image size: 10MB
        
    -   Max video duration: 5 minutes
        
-   **Frame Extraction Strategy:**
    
    -   _Scene Detection:_ Frames extracted where scene changes exceed a threshold, ensuring key moments captured.
        
    -   _Fixed FPS Sampling:_ Uniformly sample frames at ~0.67 fps for coverage across the whole video.
        
    -   _Luma Spike Detection:_ Identify and extract frames where rapid brightness changes occur, capturing sudden visual shifts.
        
-   **Deduplication:**  
    MD5 hash-based deduplication to avoid repeated frames.
    
-   **Processing Pipeline:**
    
    -   Preprocessing runs on CPU (downloads, frame extraction).
        
    -   Scanning runs on GPU (NSFW classification).
        
-   **Directory Layout:**
    
```python
/tmp/data/{round_id}/images/{job_id}.{ext}
/tmp/data/{round_id}/videos/{job_id}/frames/frame_0001.jpg
/tmp/data/{round_id}/videos/{job_id}/done.json
```
    

----------

## Usage

### Input format for `analyze_media(request: Dict)`

```json
{  "type":  "videos",  // or "images"  "jobs":  [  {"job_id":  "1",  "url":  "http://example.com/video.mp4"},  {"job_id":  "2",  "url":  "http://example.com/image.jpg"}  ]  }
``` 

### Example

```python
request = { "type": "videos", "jobs": [
        {"job_id": "123", "url": "http://example.com/sample.mp4"}
    ]
}
```

results = analyze_media(request) print(results)` 

----------

## Dependencies

-   Python 3.8+
    
-   OpenCV (`cv2`)
    
-   FFmpeg (installed on system)
    
-   `ffmpeg-python`
    
-   Custom utilities: `video_utils.py`, `nsfw_utils.py`, `helpers.py`
    

----------

## Error Handling

-   Skips unsupported or invalid media.
    
-   Gracefully handles missing frames with retry logic.
    
-   Reports errors at frame and job level.
    
-   Cleans up partial downloads or extracted data on failure.
    

----------

## Extensibility

-   Designed for batch or streaming integration.
    
-   Intermediate results can be yielded or pushed to message brokers.
    
-   Easily extendable for other detection models or media formats.
