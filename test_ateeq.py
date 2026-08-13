from pathlib import Path
from transformers import pipeline

MODEL_NAME = "Ateeqq/ai-vs-human-image-detector"
IMAGE_PATH = Path(r"C:\Users\Pc\Pictures\Saved Pictures - Copy (2)\Snapchat-1777363301.jpg")   # replace with actual path

classifier = pipeline("image-classification", model=MODEL_NAME, device=-1)
results = classifier(str(IMAGE_PATH), top_k=2)
print(results)
