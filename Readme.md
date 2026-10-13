# AI Teaching Assistant using Retrieval-Augmented Generation (RAG)

> End-to-end AI system that converts lecture videos into a searchable knowledge base and answers user questions using semantic search and Large Language Models.

## Table of Contents
1. Overview
2. Problem Statement
3. Solution
4. Objectives
5. Architecture
6. Workflow
7. Tech Stack
8. Folder Structure
9. Installation
10. Pipeline
11. Features
12. Benefits
13. Challenges
14. Future Improvements
15. License

# Overview
Educational videos contain valuable knowledge but are difficult to search. This project builds an AI Teaching Assistant that indexes lecture videos using RAG.

# Problem Statement
Students often spend hours searching videos for specific concepts. Keyword search cannot understand meaning and timestamps are hard to locate.

# Solution
The system converts videos into transcripts, splits them into chunks, generates embeddings, stores them in a vector database, retrieves relevant context, and sends that context to an LLM for grounded answers.

# Objectives
- Semantic search over videos
- Reduce search time
- Improve learning efficiency
- Minimize hallucinations using RAG
- Fast Response 

# Architecture
                    Lecture Videos
                           │
                           ▼
                  Video to MP3 Conversion
                           │
                           ▼
                  Speech-to-Text (Whisper)
                           │
                           ▼
                    Transcript JSON
                           │
                           ▼
                     Text Cleaning
                           │
                           ▼
                   Text Chunking
                           │
                           ▼
                Embedding Generation
                           │
                           ▼
             ChromaDB / Vector Database
                           │
─────────────────────────────────────────────────

                 User asks a Question
                           │
                           ▼
                  Query Embedding
                           │
                           ▼
              Similarity Search (Top-K)
                           │
                           ▼
                  Prompt Construction
                           │
                           ▼
              Large Language Model (LLM)
                           │
                           ▼
                  Context-Aware Answer

# Workflow
1. Collect videos.
2. Convert video to MP3.
3. Transcribe audio to JSON.
4. Clean and chunk text.
5. Generate embeddings.
6. Store vectors.
7. Embed user query.
8. Retrieve Top-K chunks.
9. Build prompt.
10. Generate answer.

# Technology Stack
| Layer | Technology |
|---|---|
| Language | Python |
| STT | Whisper |
| Embeddings | BGE-M3 / Nomic |
| Vector DB | ChromaDB |
| Backend | FastAPI |
| LLM | Gemini / Llama / GPT |
| Serialization | Joblib |

# Folder Structure
```text
project/
├── videos/
├── mp3/
├── transcripts/
├── embeddings/
├── scripts/
├── utils/
├── app.py
├── requirements.txt
└── README.md
```

# Installation
```bash
git clone <repo>
cd project
pip install -r requirements.txt
```

# Pipeline
## Video to MP3
Extract audio using FFmpeg/MoviePy.

## MP3 to Transcript
Use Whisper to generate timestamped JSON.

## Chunking
Split transcript into overlapping chunks.

## Embeddings
Convert chunks into vectors.

## Retrieval
Retrieve Top-K similar chunks for the user query.

## Prompt
Combine retrieved context with the question and send to the LLM.

# Features
- Video indexing
- Semantic search
- RAG
- Context-aware answers
- Modular pipeline

# Benefits
- Saves time
- Accurate answers
- Scalable
- Easy to extend to PDFs and documents

# Challenges
- Long transcript chunking
- Embedding quality
- Hallucination reduction
- Retrieval optimization

# Future Improvements
- OCR
- Multi-language support
- Timestamp citations
- Hybrid search
- Agentic RAG


