import json

import joblib
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# Configuration
# -----------------------------------
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b"

TOP_K = 5


# -----------------------------------
# Create Embeddings
# -----------------------------------

def create_embeddings(text_list):

    response = requests.post(
        OLLAMA_EMBED_URL,
        json={
            "model": EMBED_MODEL,
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
# LLM Inference
# -----------------------------------

def inference(prompt):

    response = requests.post(
        OLLAMA_GENERATE_URL,
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=600
    )

    if response.status_code != 200:
        print(response.text)
        raise Exception("LLM generation failed.")

    return response.json()


# -----------------------------------
# Load Embeddings
# -----------------------------------

print("Loading embeddings...")

df = joblib.load("embeddings.joblib")

print(f"Loaded {len(df)} chunks successfully.")


# -----------------------------------
# Question Loop
# -----------------------------------

while True:

    incoming_query = input("\nAsk a Question (type 'exit' to quit): ")

    if incoming_query.lower() == "exit":
        print("Exiting...")
        break

    # -----------------------------------
    # Query Embedding
    # -----------------------------------

    question_embedding = create_embeddings([incoming_query])[0]

    # -----------------------------------
    # Similarity Search
    # -----------------------------------

    similarities = cosine_similarity(
        np.vstack(df["embedding"].values),
        [question_embedding]
    ).flatten()

    top_indices = similarities.argsort()[::-1][:TOP_K]

    retrieved_chunks = df.iloc[top_indices]

    # -----------------------------------
    # Prompt
    # -----------------------------------

    prompt = f"""
        You are an AI assistant for a document-based RAG system.

        Below are the most relevant document chunks retrieved from the knowledge base.

        {retrieved_chunks[['title','number','start','end','text']].to_json(orient='records', indent=2)}

        -------------------------------------------------------

        User Question:

        {incoming_query}

        Instructions:

        1. Answer ONLY using the provided document chunks.
        2. Mention the document title if available.
        3. Mention the document number if available.
        4. Mention the start and end timestamps whenever possible.
        5. Give a natural, conversational answer.
        6. If the answer cannot be found in the retrieved chunks, say:
        "I couldn't find this information in the available documents."
        7. Do not make up information.
    """

    # Save Prompt
    with open("prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt)

    # -----------------------------------
    # Generate Answer
    # -----------------------------------

    response = inference(prompt)

    answer = response["response"]

    # Save Response
    with open("response.txt", "w", encoding="utf-8") as f:
        f.write(answer)

    # -----------------------------------
    # Print Retrieved Chunks
    # -----------------------------------

    print("\n")
    print("=" * 100)
    print("Retrieved Chunks")
    print("=" * 100)

    for rank, idx in enumerate(top_indices, start=1):

        row = df.iloc[idx]

        print(f"\nRank       : {rank}")
        print(f"Similarity : {similarities[idx]:.4f}")

        print(f"Title      : {row.get('title', 'N/A')}")
        print(f"Number     : {row.get('number', 'N/A')}")
        print(f"Start      : {row.get('start', 'N/A')}")
        print(f"End        : {row.get('end', 'N/A')}")

        print("\nText:")
        print(row["text"])

        print("-" * 100)

    print("\n")
    print("=" * 100)
    print("LLM Response")
    print("=" * 100)

    print(answer)