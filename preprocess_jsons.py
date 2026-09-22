import os
import json
import requests
import pandas as pd
import joblib

# -----------------------------------
# Configuration
# -----------------------------------
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text" # Change to "nomic-embed-text" if required
BATCH_SIZE = 32


def create_embeddings(text_list):
    """
    Generate embeddings for multiple texts.
    """
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "input": text_list
        },
        timeout=600
    )

    if response.status_code != 200:
        print(response.text)
        raise Exception("Embedding generation failed.")

    data = response.json()

    if "embeddings" in data:
        return data["embeddings"]

    if "embedding" in data:
        return [data["embedding"]]

    raise Exception(
        f"Unexpected response:\n{json.dumps(data, indent=4)}"
    )


# -----------------------------------
# Read all JSON files
# -----------------------------------

json_folder = "jsons"

json_files = sorted(
    [
        f
        for f in os.listdir(json_folder)
        if f.endswith(".json")
    ]
)

all_chunks = []
chunk_id = 0

for json_file in json_files:

    print(f"\nCreating Embeddings for {json_file}")

    with open(
        os.path.join(json_folder, json_file),
        "r",
        encoding="utf-8"
    ) as f:
        content = json.load(f)

    texts = [chunk["text"] for chunk in content["chunks"]]

    embeddings = []

    # -----------------------------------
    # Generate embeddings in batches
    # -----------------------------------

    for i in range(0, len(texts), BATCH_SIZE):

        batch = texts[i:i + BATCH_SIZE]

        print(
            f"Embedding Batch {i // BATCH_SIZE + 1} "
            f"({len(batch)} chunks)"
        )

        batch_embeddings = create_embeddings(batch)

        embeddings.extend(batch_embeddings)

    print(f"Chunks      : {len(texts)}")
    print(f"Embeddings  : {len(embeddings)}")

    # -----------------------------------
    # Attach embeddings to each chunk
    # -----------------------------------

    for chunk, embedding in zip(content["chunks"], embeddings):

        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embedding

        all_chunks.append(chunk)

        chunk_id += 1

    print(f"✓ Completed {json_file}")

# -----------------------------------
# Create DataFrame
# -----------------------------------

df = pd.DataFrame.from_records(all_chunks)

print("\n-----------------------------------")
print(f"Total Chunks : {len(df)}")
print("-----------------------------------")

# -----------------------------------
# Save DataFrame
# -----------------------------------

joblib.dump(df, "embeddings.joblib")

print("\n✅ Saved embeddings.joblib")
print(f"Total Records : {len(df)}")