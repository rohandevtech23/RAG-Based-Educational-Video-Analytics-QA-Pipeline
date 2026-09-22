import sys
import whisper
import json

# Configure terminal to print UTF-8 characters
sys.stdout.reconfigure(encoding="utf-8")

model = whisper.load_model("tiny")

result = model.transcribe(
    audio="audios/2_Your First HTML Website.mp3",
    language="hi",
    task="translate",
    word_timestamps=False
)

chunks = []

for segment in result["segments"]:
    chunks.append({
        "start": segment["start"],
        "end": segment["end"],
        "text": segment["text"]
    })

print(chunks)

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=4)

print("JSON file created successfully!")