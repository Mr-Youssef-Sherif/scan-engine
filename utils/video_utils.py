import os
import cv2
import ffmpeg
import hashlib
from concurrent.futures import ThreadPoolExecutor
import shutil
import time
import json

def log(msg):
    print(f"[LOG] {msg}")

# ==== Frame Extraction ====
def extract_video_frames(video_path, output_dir, fps=0.6667, scene_threshold=0.4, max_fps_frames=60, max_spike_frames=30):
    start_time = time.time()
    log(f"Starting frame extraction: {video_path}")
    os.makedirs(output_dir, exist_ok=True)
    scene_dir, fps_dir, spike_dir, final_dir = [os.path.join(output_dir, sub) for sub in ["scene", "fps", "spike", "final"]]
    for d in [scene_dir, fps_dir, spike_dir, final_dir]:
        os.makedirs(d, exist_ok=True)

    def run_ffmpeg(input_path, output_pattern, vf_filter, label):
        log(f"Running FFmpeg ({label})...")
        try:
            (
                ffmpeg
                .input(input_path)
                .output(
                    output_pattern,
                    vf=vf_filter,
                    format='image2',
                    vcodec='mjpeg',
                    q=3,
                    pix_fmt='yuvj420p',
                    threads=2,
                    vsync='vfr'
                )
                .overwrite_output()
                .run(quiet=True)
            )
            log(f"✅ FFmpeg ({label}) done.")
        except ffmpeg.Error as e:
            log(f"❌ FFmpeg error ({label}): {(e.stderr or b'').decode()}")
            raise

    def run_scene():
        run_ffmpeg(
            video_path,
            os.path.join(scene_dir, "frame_%03d.jpg"),
            f"select='gt(scene,{scene_threshold})*not(mod(n\\,5))'",
            "scene"
        )

    def run_fps():
        run_ffmpeg(
            video_path, 
            os.path.join(fps_dir, "frame_%03d.jpg"), 
            f"fps={fps},scale=320:-1",
            "fps"
        )

    def run_spike_frames():
        log("Detecting luma spikes...")
        spike_timestamps = detect_luma_spike_timestamps(video_path)
        log(f"Found {len(spike_timestamps)} spikes.")
        spike_timestamps = spike_timestamps[:max_spike_frames]
        with ThreadPoolExecutor(max_workers=4) as pool:
            pool.map(lambda i_ts: extract_at_ts(video_path, spike_dir, *i_ts), enumerate(spike_timestamps))

    def extract_at_ts(path, out_dir, i, ts):
        try:
            (
                ffmpeg
                .input(path, ss=ts)
                .output(os.path.join(out_dir, f"frame_{i:03d}.jpg"), vframes=1,
                        format='image2', vcodec='mjpeg', q=3, pix_fmt='yuvj420p',
                        vf='scale=320:-1')
                .overwrite_output()
                .run(quiet=True)
            )
            #log(f"Extracted spike frame at {ts:.2f}s -> frame_{i:03d}.jpg") # It prints per frame so group it later
        except ffmpeg.Error as e:
            log(f"❌ Error extracting spike frame at {ts}s: {(e.stderr or b'').decode()}")

    # Run extraction in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(run_scene),
            executor.submit(run_fps),
            executor.submit(run_spike_frames),
        ]
        for f in futures:
            f.result()

    # Deduplicate frames
    seen = set()
    frame_paths = []

    def add_frames(folder, cap=250):
        log(f"Adding frames from {folder} (cap={cap})...")
        count = 0
        added = 0
        for fname in sorted(os.listdir(folder)):
            if cap and count >= cap:
                break
            fpath = os.path.join(folder, fname)
            try:
                h = file_md5(fpath)
                if h not in seen:
                    seen.add(h)
                    dest = os.path.join(final_dir, f"{h}.jpg")
                    shutil.move(fpath, dest)
                    frame_paths.append(dest)
                    added += 1
                count += 1
            except Exception as e:
                log(f"❌ [Frame Error] {fpath}: {e}")
        log(f"✔️ {added} unique frames added from {folder}")

    add_frames(scene_dir)
    add_frames(fps_dir, cap=max_fps_frames)
    add_frames(spike_dir)

    # After frame extraction and deduplication
    log(f"Extraction complete. Total final frames: {len(frame_paths)}")
    elapsed = time.time() - start_time
    log(f"Total time: {elapsed:.2f}s")
    
    # Ensure all final frames exist before returning
    missing = [f for f in frame_paths if not os.path.exists(f)]
    if missing:
        log(f"⚠️ Waiting for {len(missing)} frame(s) to flush to disk...")
        for i in range(10):  # wait up to ~2.5s
            time.sleep(0.25)
            missing = [f for f in frame_paths if not os.path.exists(f)]
            if not missing:
                break
            
    if missing:
        log(f"❌ Still missing {len(missing)} frame(s): {missing[:3]}")
        raise FileNotFoundError("Some final frames were not written to disk.")
    else:
        log("✅ All final frames are confirmed on disk.")
    
    # === Write done.json as metadata marker ===
    done_path = os.path.join(output_dir, "done.json")
    done_data = {
        "total_frames": len(frame_paths),
        "output_dir": output_dir,
        "final_dir": final_dir,
        "time_elapsed": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
    with open(done_path, "w") as f:
        json.dump(done_data, f, indent=2)
    log(f"✅ Wrote metadata to {done_path}")
    
    return frame_paths

# ==== Spike Detection ====
def detect_luma_spike_timestamps(video_path, diff_thresh=15, sample_interval=0.1, suppression_window=1.0):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        cap.release()
        return []

    step = int(fps * sample_interval)
    timestamps = []
    prev_gray = None
    frame_idx = 0
    last_spike_time = -suppression_window

    while cap.isOpened():
        if frame_idx % step != 0:
            cap.grab()
            frame_idx += 1
            continue

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            delta = diff.mean()
            ts = frame_idx / fps

            if delta > diff_thresh and (ts - last_spike_time) >= suppression_window:
                timestamps.append(round(ts, 2))
                last_spike_time = ts

        prev_gray = gray
        frame_idx += 1

    cap.release()
    return timestamps


def file_md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()
