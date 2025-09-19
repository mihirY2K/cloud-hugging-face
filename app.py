import os
from typing import List
import numpy as np
from flask import Flask, request, jsonify
from pinecone import Pinecone, ServerlessSpec
import boto3
import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.huggingface import get_huggingface_llm_image_uri


PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD") or "aws"
PINECONE_REGION = os.environ.get("PINECONE_REGION") or "us-east-1"
MINILM_ENDPOINT = os.environ.get("MINILM_ENDPOINT") or "minilm-demo"
LLM_ENDPOINT = os.environ.get("LLM_ENDPOINT") or "flan-t5-demo"


role = sagemaker.get_execution_role()

llm_predictor = sagemaker.predictor.Predictor(endpoint_name=LLM_ENDPOINT)


encoder_predictor = sagemaker.predictor.Predictor(endpoint_name=MINILM_ENDPOINT)

pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
index_name = "retrieval-augmentation-aws"
index = pc.Index(index_name)


prompt_template = """Answer the following QUESTION based on the CONTEXT
given. If you do not know the answer and the CONTEXT doesn't
contain the answer truthfully say "I don't know".

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

def embed_docs(docs: List[str]) -> List[List[float]]:
    """
    Get embeddings for a list of documents using the MiniLM endpoint
    """
    out = encoder_predictor.predict({"inputs": docs})
    embeddings = np.mean(np.array(out), axis=1)
    return embeddings.tolist()

separator = "\n"
max_section_len = 1000

def construct_context(contexts: List[str]) -> str:
    chosen_sections = []
    chosen_sections_len = 0
    for text in contexts:
        text = text.strip()
        chosen_sections_len += len(text) + 2
        if chosen_sections_len > max_section_len:
            break
        chosen_sections.append(text)
    return separator.join(chosen_sections)

def rag_query(question: str) -> str:
    # Generate query embedding
    query_vec = embed_docs([question])[0]
    # Query Pinecone
    res = index.query(vector=query_vec, top_k=5, include_metadata=True)
    contexts = [match.metadata['text'] for match in res.matches]
    # Construct context string
    context_str = construct_context(contexts)
    # Build prompt
    text_input = prompt_template.replace("{context}", context_str).replace("{question}", question)
    # Query LLM
    out = llm_predictor.predict({"inputs": text_input})
    return out[0]["generated_text"]

app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        answer = rag_query(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
