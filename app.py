import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm
from getpass import getpass
from datasets import load_dataset
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

#=========================================
#Project configuration constants
#=========================================
INDEX_NAME = "context-aware-rag"
NAMESPACE = "__default__"
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMENSION = 1536
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
#Keeping it small chunks to avoid exceeding the token limit for embeddings
MAX_CHUNKS_TO_INDEX = 25
# Safe batch size for Pinecone
UPSERT_BATCH_SIZE = 50
# Keep metadata small and safe
MAX_METADATA_TEXT_CHARS = 3000

openai_client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)


wiki = load_dataset(
    "Salesforce/wikitext",
    "wikitext-2-raw-v1",
    split="train[:500]"
)
raw_text = "\n".join(
    row["text"]
    for row in wiki
    if row["text"].strip()
)
print("Total characters:", len(raw_text))
print(raw_text[:1500])


source_doc = Document(page_content=raw_text,
    metadata={
        "source": "Salesforce/wikitext",
        "subset": "wikitext-2-raw-v1",
        "doc_id": "wikitext-demo-doc"
    }
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    add_start_index=True
)
split_docs = text_splitter.split_documents([source_doc])
print("Total chunks created:", len(split_docs))

for i, doc in enumerate(split_docs[:3], start=1):
    print("\n" + "=" * 80)
    print(f"Chunk {i}")
    print("Start index:", doc.metadata.get("start_index"))
    print("-" * 80)
    print(doc.page_content[:800])


def generate_document_summary(document_text):
    preview_text = document_text[:6000]
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
You create short factual summaries for retrieval systems.
Use only the provided document.
Do not add facts that are not present.
"""
            },
            {
                "role": "user",
                "content": f"""
Create a short 3-line summary of this document.
Document:
{preview_text}
"""
            }
        ]
    )
    return response.choices[0].message.content.strip()
document_summary = generate_document_summary(raw_text)

print(document_summary)

def generate_chunk_context(document_summary, chunk_text):
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
You create short retrieval context for document chunks.
Rules:
1. Use only the provided document summary and chunk.
2. Do not add new facts.
3. Do not answer a question.
4. Do not rewrite the chunk.
5. Return 1 to 2 short factual sentences explaining what the chunk is
about.
"""
            },
            {
                "role": "user",
                "content": f"""
Document Summary:
{document_summary}
Chunk:
{chunk_text}
Return only the retrieval context.
"""
            }
        ]
    )
    return response.choices[0].message.content.strip()


def build_context_aware_text(document_summary, chunk_context, chunk_text):
    return f"""
Document Context:
{document_summary}
Chunk Context:
{chunk_context}
Chunk Content:
{chunk_text}
""".strip()
records = []
docs_to_process = split_docs
if MAX_CHUNKS_TO_INDEX is not None:
    docs_to_process = split_docs[:MAX_CHUNKS_TO_INDEX]
for chunk_id, doc in enumerate(tqdm(docs_to_process, desc="Generating chunkcontext")):
    raw_chunk = doc.page_content
    chunk_context = generate_chunk_context(
        document_summary=document_summary,
        chunk_text=raw_chunk
    )
    context_aware_text = build_context_aware_text(
        document_summary=document_summary,
        chunk_context=chunk_context,
        chunk_text=raw_chunk
    )
    record = {
        "id": f"{doc.metadata['doc_id']}-chunk-{chunk_id}",
        "text_to_embed": context_aware_text,
        "metadata": {
            "doc_id": doc.metadata.get("doc_id"),
            "chunk_id": chunk_id,
            "source": doc.metadata.get("source"),
            "subset": doc.metadata.get("subset"),
            "start_index": doc.metadata.get("start_index"),
            "chunk_context": chunk_context[:800],
            "raw_chunk": raw_chunk[:MAX_METADATA_TEXT_CHARS]
        }
    }
    records.append(record)
print("Total context-aware records:", len(records))


trace_df = pd.DataFrame([
    {
        "record_id": r["id"],
        "chunk_id": r["metadata"]["chunk_id"],
        "start_index": r["metadata"]["start_index"],
        "context_length": len(r["metadata"]["chunk_context"]),
        "raw_chunk_length": len(r["metadata"]["raw_chunk"]),
        "text_to_embed_length": len(r["text_to_embed"]),
        "context_preview": r["metadata"]["chunk_context"][:120],
        "chunk_preview": r["metadata"]["raw_chunk"] [:120].replace("\n", " ")
    }
    for r in records
])
trace_df.head(10)

sample = records[0]
print("=" * 80)
print("RAW CHUNK")
print("=" * 80)
print(sample["metadata"]["raw_chunk"][:1000])
print("\n\n" + "=" * 80)
print("LLM-GENERATED CHUNK CONTEXT")
print("=" * 80)
print(sample["metadata"]["chunk_context"])
print("\n\n" + "=" * 80)
print("CONTEXT-AWARE TEXT TO EMBED")
print("=" * 80)
print(sample["text_to_embed"][:1500])


existing_indexes = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Index created {INDEX_NAME}")
else:
    print("Index already exists.")
index = pc.Index(INDEX_NAME)


def embed_texts(texts):
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]

def upsert_records_to_pinecone(records, batch_size=UPSERT_BATCH_SIZE):
    total_upserted = 0
    for start in tqdm(range(0, len(records), batch_size), desc="Embedding + Upserting"):
        batch = records[start:start + batch_size]
        texts = [r["text_to_embed"] for r in batch]
        vectors = embed_texts(texts)
        pinecone_vectors = []
        for record, vector in zip(batch, vectors):
            pinecone_vectors.append({
                "id": record["id"],
                "values": vector,
                "metadata": record["metadata"]
            })
        index.upsert(
            vectors=pinecone_vectors,
            namespace=NAMESPACE
        )
        total_upserted += len(pinecone_vectors)
    return total_upserted


total_upserted = upsert_records_to_pinecone(records)
print("Total records inserted into Pinecone:", total_upserted)

stats = index.describe_index_stats()


def retrieve_context(query, top_k=5):
    query_vector = embed_texts([query])[0]
    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
        namespace=NAMESPACE
    )
    return response

query = "what is Valkyria Chronicles III ? "
retrieval_response = retrieve_context(query, top_k=5)
for i, match in enumerate(retrieval_response.matches, start=1):
    print("\n" + "=" * 80)
    print(f"Result {i}")
    print("Score:", match.score)
    print("Record ID:", match.id)
    print("Chunk ID:", match.metadata["chunk_id"])
    print("Source:", match.metadata["source"])
    print("Start Index:", match.metadata["start_index"])
    print("\nChunk Context:")
    print(match.metadata["chunk_context"])
    print("\nRaw Chunk:")
    print(match.metadata["raw_chunk"][:700])


def build_rag_context(matches):
    context_blocks = []
    for i, match in enumerate(matches, start=1):
        metadata = match.metadata
        block = f"""
Source {i}
Score: {match.score}
Document ID: {metadata.get("doc_id")}
Chunk ID: {metadata.get("chunk_id")}
Source: {metadata.get("source")}
Start Index: {metadata.get("start_index")}
Chunk Context:
{metadata.get("chunk_context")}
Raw Chunk:
{metadata.get("raw_chunk")}
"""
        context_blocks.append(block.strip())
    return "\n\n---\n\n".join(context_blocks)


def answer_with_rag(question, top_k=5):
    retrieval_response = retrieve_context(question, top_k=top_k)
    if not retrieval_response.matches:
        return {
            "answer": "No relevant chunks found.",
            "sources": []
        }
    rag_context = build_rag_context(retrieval_response.matches)
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
You are a careful RAG assistant.
Rules:
1. Answer only using the provided context.
2. If the context is not enough, say the available context is insufficient.
3. Do not invent facts.
4. Mention relevant source/chunk IDs when helpful.
"""
            },
            {
                "role": "user",
                "content": f"""
Retrieved Context:
{rag_context}
Question:
{question}
"""
            }
        ]
    )
    answer = response.choices[0].message.content.strip()
    sources = [
        {
            "record_id": match.id,
            "score": match.score,
            "chunk_id": match.metadata.get("chunk_id"),
            "source": match.metadata.get("source"),
            "start_index": match.metadata.get("start_index"),
            "chunk_context": match.metadata.get("chunk_context")
        }
        for match in retrieval_response.matches
    ]
    return {
        "answer": answer,
        "sources": sources
    }


result = answer_with_rag(
    question="What is the main topic discussed in the retrieved document?",
    top_k=5
)
print("=" * 80)
print("FINAL RAG ANSWER")
print("=" * 80)
print(result["answer"])


questions = [
    "What is the document mainly about?",
    "Summarize the key information from the retrieved chunks.",
    "What named entities or events are mentioned?",
    "What is the historical or factual context discussed?"
]
for q in questions:
    print("\n" + "=" * 100)
    print("QUESTION:", q)
    print("=" * 100)
    result = answer_with_rag(q, top_k=5)
    print(result["answer"])