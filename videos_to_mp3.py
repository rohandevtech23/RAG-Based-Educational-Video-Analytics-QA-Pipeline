import os
import re
import subprocess

os.makedirs("audios", exist_ok=True)

files = os.listdir("videos")

for file in files:

    tutorial_number = file.split(" #")[1].split(" ")[0]

    file_name = file.split(" ｜ ")[0]

    safe_name = re.sub(r'[<>:"/\\|?*]', "", file_name)

    output = os.path.join("audios", f"{tutorial_number}_{safe_name}.mp3")

    print(output)

    subprocess.run([
        "ffmpeg",
        "-i",
        os.path.join("videos", file),
        output
    ])

       