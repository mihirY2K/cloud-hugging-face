RAG Flask API with SageMaker and Pinecone
Overview

This project implements a Retrieval-Augmented Generation (RAG) API using Flask, AWS SageMaker, and Pinecone.
It allows users to query a large knowledge base and get context-aware answers using a combination of document embeddings (MiniLM) and LLM inference (Flan-T5).

The pipeline works as follows:

A user submits a question to the API.

The question is converted into an embedding using MiniLM.

The embedding is used to query a Pinecone vector database for the most relevant documents.

Retrieved documents are combined into a prompt with the question.

The prompt is sent to a Flan-T5 endpoint to generate a context-aware answer.

The API returns the answer as JSON.

Features

Real-time question-answering over a large custom knowledge base

Uses Pinecone for scalable vector search

Integrates with AWS SageMaker for both embedding and LLM endpoints

Flask API for easy deployment and integration
