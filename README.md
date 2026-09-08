# DocuRAG — Intelligent PDF Analysis & Hybrid RAG

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![RAG](https://img.shields.io/badge/AI-RAG-purple)
![Gemini](https://img.shields.io/badge/LLM-Gemini-blue)
![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-green)
![Sentence Transformers](https://img.shields.io/badge/Embeddings-Sentence%20Transformers-orange)

DocuRAG is a document-intelligence application that allows users to upload a PDF, receive a structured summary, and ask questions grounded in the document's actual content.

The application retrieves relevant text from the uploaded PDF before generating an answer and returns page-level source information alongside PDF-based responses. When a question genuinely requires information beyond the uploaded document, DocuRAG can use web retrieval and keeps external information explicitly separate from PDF evidence.

This is a learning and portfolio project built to demonstrate an explicit end-to-end Retrieval-Augmented Generation (RAG) pipeline without hiding the core retrieval logic behind a framework abstraction.

The pipeline includes:

* PDF text extraction
* Metadata extraction
* Text cleaning
* Page-aware chunking
* Local embeddings
* FAISS vector indexing
* Semantic retrieval
* Grounded LLM generation
* Follow-up question handling
* Hybrid PDF and web retrieval
* Source attribution
* Basic retrieval evaluation

The project deliberately avoids abstracting away the important RAG stages so that each component can be inspected and understood independently.

---

## Table of Contents

* [Problem Statement](#problem-statement)
* [Project Goals](#project-goals)
* [Features](#features)
* [Design Principles](#design-principles)
* [Architecture](#architecture)
* [How the RAG Pipeline Works](#how-the-rag-pipeline-works)
* [Document Processing Flow](#document-processing-flow)
* [Question Answering Flow](#question-answering-flow)
* [Hybrid Retrieval](#hybrid-retrieval)
* [Technology Stack](#technology-stack)
* [Project Structure](#project-structure)
* [Module Responsibilities](#module-responsibilities)
* [Installation](#installation)
* [Prerequisites](#prerequisites)
* [API Key Configuration](#api-key-configuration)
* [Usage](#usage)
* [Screenshots](#screenshots)
* [Evaluation](#evaluation)
* [Limitations](#limitations)
* [Future Improvements](#future-improvements)

---

# Problem Statement

General-purpose LLMs do not automatically know the contents of a PDF uploaded during a user session.

If asked questions about a document without being provided relevant context, an LLM may generate a fluent answer that sounds plausible but is not actually supported by the document.

This creates a common document-Q&A problem:

```text
User uploads PDF
       ↓
User asks question
       ↓
LLM does not actually know the document
       ↓
Possible fabricated answer
```

Separately, manually reading a long document to:

* find a specific fact,
* identify the methodology,
* understand the objective,
* locate key findings,
* or create a structured summary,

can be slow and repetitive.

DocuRAG addresses this by retrieving relevant text from the uploaded document before generating a PDF-based answer.

The intended flow is:

```text
Question
   ↓
Retrieve relevant document chunks
   ↓
Provide retrieved context to LLM
   ↓
Generate answer grounded in that context
   ↓
Return answer with source pages
```

If relevant information cannot be retrieved from the document, the system is designed to avoid generating a speculative PDF answer.

---

# Project Goals

DocuRAG was designed around the following goals.

### 1. Make the RAG pipeline explicit

The project does not hide chunking, embedding, indexing, retrieval, or prompt construction behind a high-level chain abstraction.

Each major stage has its own implementation module.

### 2. Keep embeddings local

The embedding model runs locally using Sentence Transformers.

This means PDF chunks do not need to be sent to a paid embedding API.

### 3. Preserve page-level source information

Chunks retain page information so retrieved evidence can be associated with the pages from which it came.

### 4. Separate retrieval from generation

Retrieval determines what information is available to the LLM.

Generation is handled separately by `llm.py`.

This separation allows:

```text
PDF processing
     ↓
Retrieval
     ↓
Context construction
     ↓
Generation
```

instead of combining everything into a single opaque function.

### 5. Avoid pretending evaluation is more comprehensive than it is

The project includes a small evaluation harness, but its limitations are explicitly documented.

The evaluation does not claim to prove general RAG quality or universal answer correctness.

---

# Features

* **Upload a PDF** through a Streamlit interface.

* **Automatic document metadata extraction**, including information such as:

  * title,
  * authors,
  * page count,
  * document type.

* **Automatic structured summary** generated after upload, covering:

  * title,
  * authors,
  * page count,
  * document type,
  * objective,
  * problem statement,
  * methodology,
  * key findings,
  * limitations,
  * conclusion,
  * keywords.

* **Grounded PDF Q&A** using text retrieved from the uploaded document.

* **Page-level source citations** for PDF-based answers.

* **Local semantic search** using:

  * Sentence Transformers,
  * FAISS,
  * cosine similarity through normalized embeddings and inner-product search.

* **Hybrid retrieval** where a query can be routed to:

  * `PDF_ONLY`
  * `WEB_ONLY`
  * `PDF_AND_WEB`

* **Clear source separation** for hybrid answers:

```text
PDF Evidence:
...

External Information:
...
```

* **Conversation-aware follow-ups** where recent conversation history can help resolve a follow-up question into a standalone retrievable query.

* **Fresh retrieval for answers** rather than generating an answer directly from a previous response.

* **Hallucination control** by avoiding PDF-answer generation when relevant document information was not retrieved.

* **Basic evaluation harness** with:

  * Hit@K,
  * Recall@K,
  * trap-question decline behavior,
  * optional page-overlap checking for generated answers.

---

# Design Principles

## Explicit over hidden abstractions

The project is intended to demonstrate how a RAG system actually works internally.

Instead of:

```text
Document
   ↓
Black-box framework
   ↓
Answer
```

DocuRAG exposes the intermediate stages:

```text
Document
   ↓
Text Extraction
   ↓
Cleaning
   ↓
Chunking
   ↓
Embedding
   ↓
Vector Index
   ↓
Retrieval
   ↓
Context Construction
   ↓
LLM Generation
   ↓
Answer + Sources
```

---

## Retrieval before generation

For document questions, the retrieval stage is responsible for finding relevant evidence before the LLM generates an answer.

The LLM is not intended to independently "know" the uploaded PDF.

Instead:

```text
PDF
 ↓
Relevant chunks retrieved
 ↓
Chunks passed as context
 ↓
LLM generates answer
```

---

## Page information is preserved

The system processes the PDF page by page before chunking.

This allows chunks to retain page-related metadata.

That metadata can later be used for source attribution.

Conceptually:

```text
Chunk 1
Text: "..."
Page: 2

Chunk 2
Text: "..."
Page: 3
```

When chunks are retrieved, the corresponding page information can be returned with the answer.

---

## Generation is separate from RAG logic

`llm.py` intentionally does not contain RAG-specific logic.

It acts as a generic Gemini wrapper.

The RAG-specific modules determine:

* what context to retrieve,
* how to construct prompts,
* how grounding rules should work,
* when generation should happen.

This separation makes the LLM wrapper reusable for both:

```text
Structured Summary
        ↓
      llm.py
```

and:

```text
Retrieved PDF Context
        ↓
   rag_pipeline.py
        ↓
      llm.py
```

---

# Architecture

```text
                              ┌──────────────────┐
                              │    PDF Upload    │
                              └────────┬─────────┘
                                       │
                                       ▼
                         ┌──────────────────────────┐
                         │       PyMuPDF            │
                         │ Page-by-page extraction  │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ Metadata Extraction      │
                         │ Title / Authors / Type   │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ Cleaning + Boilerplate   │
                         │ Header/Footer Removal    │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ Page-aware Chunking      │
                         │ with Overlap             │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ Sentence Transformers    │
                         │ Local Embeddings         │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │       FAISS Index        │
                         │ Cosine Similarity Search │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                               Ready for Q&A


User Question
     │
     ▼
┌──────────────────────────┐
│ Follow-up Resolution     │
│ if conversation-dependent│
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Query Router             │
│ PDF / Web / Both         │
└────────────┬─────────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
┌───────────┐  ┌──────────────┐
│ FAISS PDF │  │ Tavily Search│
│ Retrieval │  │ Web Retrieval│
└─────┬─────┘  └──────┬───────┘
      │               │
      └───────┬───────┘
              │
              ▼
┌──────────────────────────┐
│ Context Assembly         │
│ Page / Source Tags       │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Gemini Generation        │
└────────────┬─────────────┘
             │
             ▼
       Answer + Sources
```

---

# How the RAG Pipeline Works

## 1. PDF text extraction

The uploaded document is processed page by page using PyMuPDF.

Conceptually:

```text
PDF
 ↓
Page 1 → text
Page 2 → text
Page 3 → text
...
```

Processing pages separately is important because page information can later be associated with retrieved chunks.

---

## 2. Metadata extraction

`metadata_extractor.py` performs heuristic extraction of document information.

The extracted information can include:

* title,
* authors,
* headings,
* document type,
* page count.

This metadata is available even when LLM generation is unavailable.

---

## 3. Text cleaning

Before embedding, document text goes through light cleaning.

The documented pipeline also includes boilerplate handling such as header and footer removal.

This helps reduce the chance that repeated document boilerplate becomes unnecessarily prominent in retrieval.

---

## 4. Page-aware chunking

A full PDF cannot simply be sent directly to a retrieval system as one block of text.

The document is divided into smaller chunks.

```text
Document

Chunk 1
Chunk 2
Chunk 3
Chunk 4
...
```

DocuRAG uses page-aware chunking with overlap.

The overlap helps preserve context between nearby chunks.

At the same time, chunks do not cross page boundaries in the current implementation, which is documented as a limitation.

---

## 5. Local embeddings

Each text chunk is converted into a vector representation using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Conceptually:

```text
"Machine learning model..."

          ↓

[0.12, -0.44, 0.81, ...]
```

The same embedding model is used to convert user queries into vectors.

This allows semantic similarity search.

---

## 6. FAISS indexing

The chunk embeddings are added to a FAISS vector index.

The current architecture uses normalized embeddings with inner-product search to represent cosine similarity.

Conceptually:

```text
PDF Chunks
    ↓
Embeddings
    ↓
FAISS Index
```

The vector index is not a trained model.

It is rebuilt from the uploaded document's chunks when that document is processed.

---

## 7. Query embedding

When a user asks a question:

```text
"What methodology was used?"
```

the question is embedded using the same embedding model.

```text
Question
   ↓
Embedding
   ↓
Query Vector
```

---

## 8. Semantic retrieval

The query vector is searched against the FAISS index.

The system retrieves the most similar chunks.

```text
Question
   ↓
FAISS Search
   ↓

Chunk 8  → Page 4
Chunk 11 → Page 5
Chunk 7  → Page 4
```

These retrieved chunks become candidate evidence for answering the question.

---

## 9. Context construction

Retrieved chunks are assembled into context for the LLM.

The context includes source information such as page references.

Conceptually:

```text
[PDF - Page 4]
Retrieved document text...

[PDF - Page 5]
Retrieved document text...
```

---

## 10. Grounded generation

Gemini receives:

```text
System Instructions
       +
Retrieved Context
       +
User Question
```

and generates the final response.

The RAG-specific grounding instructions are handled outside the generic Gemini wrapper.

---

# Document Processing Flow

When a user uploads a PDF, the application follows the document-processing pipeline.

```text
Upload PDF
     ↓
Extract page-by-page text
     ↓
Extract metadata
     ↓
Clean document text
     ↓
Create page-aware chunks
     ↓
Generate embeddings
     ↓
Build FAISS index
     ↓
Generate structured summary
     ↓
Display document information
     ↓
Enable question answering
```

The same processed document data supports both:

1. document summarization, and
2. later semantic retrieval for Q&A.

---

# Question Answering Flow

For a PDF-based question:

```text
User Question
      ↓
Embed Question
      ↓
Search FAISS
      ↓
Retrieve Relevant Chunks
      ↓
Relevant Information Found?
      │
 ┌────┴────┐
 │         │
Yes        No
 │         │
 ▼         ▼
Build      Return
Context    Not Found
 │
 ▼
Gemini
 │
 ▼
Answer + Page Sources
```

The key principle is that generation should follow retrieval rather than replacing it.

---

# Conversation-Aware Follow-ups

Users often ask questions that depend on a previous question.

For example:

```text
User:
What dataset was used?

Assistant:
The document used ...

User:
How large was it?
```

The second question may not be meaningful on its own.

DocuRAG can use recent conversation history to resolve such a question into a more standalone form suitable for retrieval.

Conceptually:

```text
Recent Conversation
       +
Current Follow-up
       ↓
Standalone Question
       ↓
Fresh Retrieval
       ↓
Answer
```

The final answer is still based on newly retrieved evidence rather than simply repeating or trusting the previous answer.

---

# Hybrid Retrieval

Some questions may require information outside the uploaded document.

The query router can classify a question as:

```text
PDF_ONLY
```

Use the uploaded document.

```text
WEB_ONLY
```

Use external web retrieval.

```text
PDF_AND_WEB
```

Use both sources.

---

## PDF-only example

```text
"What methodology does this paper use?"
```

Expected source:

```text
Uploaded PDF
```

---

## Web-only example

A question that explicitly requires current external information may be routed to web retrieval.

---

## Hybrid example

A question may require:

* evidence from the uploaded document, and
* additional external information.

For these cases, the response keeps the evidence separated.

```text
PDF Evidence:
Information retrieved from the uploaded document.

External Information:
Information retrieved from web sources.
```

This avoids presenting external information as though it came from the PDF.

---

# Hallucination Control

DocuRAG does not treat the LLM as a replacement for document retrieval.

For PDF-related questions, the intended sequence is:

```text
Retrieve Evidence
      ↓
Generate Answer
```

not:

```text
Ask LLM
   ↓
Hope Answer Is Correct
```

The documented implementation also avoids making an LLM call when nothing relevant was retrieved for a PDF question.

Instead, the application can indicate that the information could not be found in the uploaded document.

This distinction is important because:

```text
No answer found
```

is preferable to:

```text
Confident but unsupported answer
```

---

# Technology Stack

| Purpose                   | Choice                                   | Role in Project                                                     |
| ------------------------- | ---------------------------------------- | ------------------------------------------------------------------- |
| UI                        | Streamlit                                | Builds the interactive PDF analysis and Q&A interface               |
| PDF parsing               | PyMuPDF                                  | Extracts PDF content page by page                                   |
| Metadata extraction       | Custom heuristics                        | Extracts title, authors, headings, and related document information |
| Chunking                  | Custom implementation                    | Performs cleaning and page-aware chunking                           |
| Embeddings                | `sentence-transformers/all-MiniLM-L6-v2` | Converts document chunks and queries into vectors                   |
| Vector search             | FAISS (`faiss-cpu`)                      | Performs local similarity search                                    |
| LLM                       | Gemini via `google-genai`                | Generates summaries and grounded answers                            |
| Web retrieval             | Tavily                                   | Retrieves external information when web retrieval is used           |
| Environment configuration | `python-dotenv`                          | Loads API keys from environment configuration                       |
| Evaluation                | Custom implementation                    | Calculates retrieval and basic answer-behavior metrics              |
| Notebook                  | Jupyter                                  | Provides an educational step-by-step RAG walkthrough                |

---

# Why No LangChain?

This project deliberately does not use LangChain.

That is not because LangChain is inherently unsuitable for RAG systems.

The choice is educational and architectural.

The goal is to make each stage visible:

```text
Chunking
Embedding
Indexing
Retrieval
Context Formatting
Prompt Construction
Generation
Source Attribution
```

Using direct implementations makes it easier to inspect:

* what enters the vector store,
* what is retrieved,
* what context reaches the LLM,
* and where a failure or poor answer originates.

For a learning and portfolio project, this provides a clearer demonstration of the underlying RAG architecture.

---

# Project Structure

```text
docurag/

├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
│
├── data/
│   ├── documents/
│   │   └── sample_paper.pdf
│   │
│   └── evaluation/
│       └── eval_set.json
│
├── notebooks/
│   └── document_rag_experiments.ipynb
│
├── src/
│   ├── pdf_loader.py
│   ├── metadata_extractor.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── web_retriever.py
│   ├── query_router.py
│   ├── llm.py
│   ├── summarizer.py
│   ├── rag_pipeline.py
│   └── evaluation.py
│
└── vectorstore/
```

The `vectorstore/` directory contains generated runtime data and is gitignored.

---

# Module Responsibilities

## `app.py`

The Streamlit application entry point.

Responsible for connecting the interface with the document-processing and question-answering pipeline.

---

## `pdf_loader.py`

Responsible for:

```text
PDF
 ↓
Page-by-page text extraction
```

---

## `metadata_extractor.py`

Responsible for heuristic extraction of document information such as:

* title,
* authors,
* headings.

---

## `chunker.py`

Responsible for:

* text cleaning,
* boilerplate handling,
* page-aware chunking,
* overlapping chunks.

---

## `embeddings.py`

Responsible for:

* loading the embedding model,
* converting text into vectors.

---

## `vector_store.py`

Responsible for:

* creating the FAISS index,
* storing the mapping between vectors and chunk metadata.

---

## `retriever.py`

Responsible for:

* embedding user queries,
* searching the PDF vector index,
* returning relevant document chunks,
* formatting PDF context and sources.

---

## `web_retriever.py`

Responsible for:

* performing Tavily web retrieval,
* formatting retrieved external information,
* preparing external source information.

---

## `query_router.py`

Responsible for routing questions into:

```text
PDF_ONLY
WEB_ONLY
PDF_AND_WEB
```

The current routing approach is heuristic and phrase-based rather than learned.

---

## `llm.py`

A generic Gemini API wrapper.

It intentionally does not contain:

* retrieval logic,
* PDF-specific grounding logic,
* query routing.

This keeps generation separate from the modules that determine what information should be sent to the LLM.

---

## `summarizer.py`

Responsible for generating the structured document summary.

The documented implementation uses a map-reduce approach for summarization.

---

## `rag_pipeline.py`

Coordinates the main RAG flow.

Conceptually:

```text
Question
   ↓
Retrieval
   ↓
Context
   ↓
Generation
   ↓
Sources
```

---

## `evaluation.py`

Runs the project's small evaluation harness.

Responsible for calculating and reporting the documented evaluation metrics.

---

# Installation

## Prerequisites

Before running the project, you need:

* Python
* `pip`
* a Gemini API key for summarization and LLM answer generation
* optionally, a Tavily API key for web retrieval

---

## Clone the repository

```bash
git clone <this-repo-url>
cd docurag
```

---

## Create a virtual environment

```bash
python -m venv .venv
```

---

## Activate the environment

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

The first time the embedding model is used, Sentence Transformers downloads:

```text
sentence-transformers/all-MiniLM-L6-v2
```

and caches it locally.

This initial model download requires internet access.

After the model has been downloaded and cached, embedding inference itself runs locally.

---

# API Key Configuration

Create a `.env` file using `.env.example` as a reference.

Example:

```text
GEMINI_API_KEY=your_key_here

TAVILY_API_KEY=your_key_here
```

---

## Gemini API key

Required for:

* AI-generated structured summaries,
* LLM-generated answers.

Without `GEMINI_API_KEY`, the application can still access the non-LLM document processing described in the project, such as basic metadata extraction, but it cannot generate summaries or LLM answers.

---

## Tavily API key

Optional.

Required only when a question is routed to web retrieval.

Without `TAVILY_API_KEY`:

* PDF-only functionality remains available.
* Web retrieval is unavailable.
* Questions requiring web retrieval should indicate that web search is not configured.

---

## Important security note

Never commit:

```text
.env
```

to GitHub.

API keys should remain outside source control.

The repository includes `.env` in `.gitignore`.

---

# Usage

## Run the Streamlit application

```bash
streamlit run app.py
```

The application opens in your browser.

---

## Typical workflow

### Step 1 — Upload a PDF

The application processes the uploaded document.

---

### Step 2 — View document information

The interface displays available structured document information such as:

* title,
* authors,
* page count,
* document type.

---

### Step 3 — Generate summary

If Gemini is configured and available, the application generates the structured summary.

---

### Step 4 — Ask questions

Examples:

```text
What is the main objective of this document?
```

```text
What methodology was used?
```

```text
What dataset was used?
```

```text
What limitations were identified?
```

The system retrieves relevant chunks before generating a document-grounded answer.

---

## Run the experiments notebook

The notebook provides a step-by-step educational walkthrough of the RAG pipeline.

```bash
jupyter notebook notebooks/document_rag_experiments.ipynb
```

The notebook demonstrates the RAG stages independently before they are combined into the application.

---

## Run the evaluation harness

Use the bundled evaluation resources:

```bash
python src/evaluation.py
```

Or provide a PDF and evaluation set:

```bash
python src/evaluation.py path/to/your.pdf your_eval_set.json
```

---

# Screenshots

Add screenshots after running the application locally.

Recommended screenshots include:

```text
assets/
└── screenshots/
    ├── summary_panel.png
    └── chat_panel.png
```

Then reference them in this README:

```markdown
![Document summary panel](assets/screenshots/summary_panel.png)

![Chat with sources](assets/screenshots/chat_panel.png)
```

Useful screenshots would demonstrate:

1. PDF upload
2. Structured document metadata
3. Generated summary
4. PDF question answering
5. Page-level source information
6. Hybrid answer source separation, if demonstrated

---

# Evaluation

`src/evaluation.py` uses a small hand-built evaluation set.

The documented evaluation includes:

* 9 questions,
* 7 answerable questions,
* 2 deliberate trap questions.

The answerable questions are spread across the sample document.

The trap questions intentionally ask for information that the document does not contain.

---

## Hit@K / Recall@K

These are retrieval metrics.

They answer a question such as:

> Did the expected relevant information appear among the retrieved chunks?

For example:

```text
Expected Page: 4

Retrieved Results:
Page 2
Page 4
Page 5
```

The retrieval successfully returned evidence from the expected page.

These metrics evaluate retrieval behavior.

They do **not** directly evaluate:

* answer wording,
* answer quality,
* factual correctness of generated text.

---

## Trap-question decline rate

The evaluation also checks whether the system appropriately declines to invent an answer when the relevant information is not in the document.

Example:

```text
Question:
What is the author's favorite programming language?

Document:
Does not contain this information.
```

The desired behavior is:

```text
Information not found in the uploaded document.
```

rather than a fabricated answer.

---

## Optional LLM-answer check

When `GEMINI_API_KEY` is configured, generated answers can also be checked against expected page overlap.

This is a coarse faithfulness proxy.

For example:

```text
Expected Evidence:
Page 4

Answer Sources:
Page 4
Page 5
```

This suggests source overlap.

However, it does **not** prove that:

* the answer is fully correct,
* every statement is supported,
* the answer interpreted the evidence correctly.

---

## What the evaluation does not prove

The evaluation is intentionally small and limited.

Its results do not prove:

* general RAG performance,
* universal factual correctness,
* full answer faithfulness,
* performance across many document types,
* performance across large document collections.

The current evaluation set is built around one sample document.

A stronger evaluation would require:

* more documents,
* more question types,
* human review,
* systematic relevance judgments,
* or an additional faithfulness evaluation approach.

---

# Limitations

## Single document at a time

The current implementation processes one uploaded document for the active session.

Uploading a new document replaces the current session's index.

There is no persisted multi-document knowledge base.

---

## No OCR

Scanned or image-only PDFs without extractable text cannot be processed by the current implementation.

The application should provide a clear error instead of silently returning an empty document.

---

## Heuristic query routing

`query_router.py` uses heuristic phrase matching.

It is deliberately simple and explainable, but it is not a learned classifier.

This means unusually phrased questions may be routed incorrectly.

---

## Page-boundary chunking

Chunks do not cross page boundaries.

A fact split between:

```text
End of Page 4
       +
Start of Page 5
```

may not be completely represented inside a single chunk.

---

## Small evaluation set

The evaluation data is hand-built for one sample document.

Its results describe retrieval behavior on that evaluation setup and should not be interpreted as general RAG performance.

---

## External service availability

Gemini generation and Tavily web retrieval depend on external API availability and account configuration.

Temporary service errors or changes in free-tier availability can affect those components.

---

## No automated test suite

The current repository does not include a committed `pytest` test suite.

The modules have been manually tested during development, but automated regression testing has not yet been added.

---

# Future Improvements

## Centralized configuration

Move configurable values into:

```text
src/config.py
```

Examples include:

* chunk size,
* chunk overlap,
* `TOP_K`,
* embedding model name,
* LLM model name.

---

## Multi-document RAG

Extend the current single-document flow into a persisted multi-document knowledge base.

Conceptually:

```text
Document A
Document B
Document C
     ↓
Combined / Managed Index
     ↓
Cross-document Retrieval
```

---

## OCR support

Add OCR fallback for scanned or image-based PDFs.

---

## Improved query routing

Evaluate the heuristic router and consider replacing it with an LLM-based classifier if the heuristic approach demonstrates frequent misrouting.

---

## Larger evaluation

Expand the evaluation across:

* multiple PDFs,
* different document lengths,
* different domains,
* answerable questions,
* unanswerable questions,
* ambiguous questions,
* hybrid retrieval questions.

---

## Stronger answer-faithfulness evaluation

The current page-overlap check is only a coarse proxy.

Future work could include:

* human evaluation,
* more systematic answer verification,
* LLM-as-judge evaluation.

---

## Automated tests

Add a `pytest` suite for modules in `src/`.

Potential coverage areas include:

* PDF extraction,
* metadata extraction,
* chunking,
* vector indexing,
* retrieval,
* query routing,
* source formatting.

---

## Deployment

Deploy the application after the local implementation and testing are sufficiently stable.

A deployment target can be added as a future stage rather than presenting deployment as already completed.

---

# Key Learning Outcomes

This project demonstrates practical understanding of:

* Retrieval-Augmented Generation
* PDF document processing
* Text chunking
* Semantic embeddings
* Vector search
* FAISS
* Cosine similarity retrieval
* Sentence Transformers
* Grounded generation
* Source attribution
* LLM API integration
* Hybrid retrieval
* Query routing
* Conversation-aware follow-ups
* RAG evaluation
* Streamlit application development
* Environment and API key management

---

# Important Project Scope Note

DocuRAG is an application and learning project.

It does **not** train a custom LLM or embedding model.

The project uses pre-trained components for inference:

```text
Sentence Transformer
        ↓
Local Embedding Inference

Gemini
        ↓
Text Generation
```

FAISS is used as a vector index for similarity search.

The vector index itself is not a trained machine learning model.

---

# Future Repository Improvements

As the project matures, the repository itself can also be improved with:

* architecture diagram image,
* application screenshots,
* automated tests,
* example evaluation outputs,
* sample public/synthetic documents,
* deployment instructions.
  
---
