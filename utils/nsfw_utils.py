import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageClassification

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "Falconsai/nsfw_image_detection"

# lazy load
_model = None
_processor = None

def get_model():
    global _model, _processor
    if _model is None or _processor is None:
        _processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
        _model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
        _model.eval()
    return _processor, _model

def scan_images_for_nsfw(image_paths, batch_size=32):
    if not image_paths:
        return []

    processor, model = get_model()
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        for j, path in enumerate(batch_paths):
            score_dict = {label: float(probs[j][k]) for k, label in model.config.id2label.items()}
            nsfw_score = score_dict.get("nsfw", 0.0)
            is_nsfw = nsfw_score > 0.5
            results.append((path, is_nsfw, nsfw_score, score_dict))

    return results
