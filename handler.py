# handler.py

from main import analyze_media

def handler(event):
    """
    RunPod entrypoint. 'event' is a dictionary containing the input payload.
    """
    try:
        print("[HANDLER] Received event:", event)
        result = analyze_media(event)
        return {
            "status": "success",
            "results": result
        }
    except Exception as e:
        print(f"[HANDLER] Error: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
