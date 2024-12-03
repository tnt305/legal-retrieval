# Legal Document Retrieval Solution for SOICT

## Overview

This solution provides a robust and efficient system for **Legal Document Retrieval** that addresses the complexities of processing legal terms and leverages **Sentence Transformer** models for enhanced semantic understanding. The primary focus is on specialized text processing techniques tailored for legal language and fine-tuning Sentence Transformer models to achieve high accuracy in legal information retrieval tasks.

---

## Features

- **Legal Term Processing**:
  - Handles long, jargon-heavy sentences common in legal documents.
  - Normalizes legal terms and abbreviations for consistent processing.
  - Supports multilingual legal texts for global applications.

- **Semantic Search**:
  - Uses Sentence Transformer models fine-tuned on legal datasets.
  - Provides accurate retrieval of legal clauses, statutes, and case laws based on user queries.

- **Fine-tuning**:
  - Enhances pre-trained Sentence Transformer models to specialize in legal document retrieval.
  - Employs pairwise ranking and classification to improve query-document relevance.

- **Customizable Workflow**:
  - Modular components allow integration with existing document management systems.
  - Configurable processing pipelines to handle domain-specific requirements.

---

## Components

### 1. Data Preprocessing

- **Legal Term Normalization**:
- **Tokenization**:
- **Stopword Handling**:
- **Named Entity Recognition (NER)**:

---

### 2. Model Fine-Tuning

- **Dataset**:
  - Prepares a legal corpus with annotated query-document pairs.
  - Example structure:
    - **Queries**: Short legal questions or issues.
    - **Documents**: Relevant statutes, precedents, or legal provisions.

- **Training Objective**:
  - Minimizes ranking loss to align query-document embeddings.

- **Evaluation Metrics**:
  - Mean Reciprocal Rank (MRR).
  - Normalized Discounted Cumulative Gain (nDCG).
  - Precision@k.

---

### 3. Search Pipeline

- **BM25s for Initial Retrieval**:
  - Quickly identifies a candidate set of documents using lexical similarity.
  
- **Semantic Reranking**:
  - Refines results using fine-tuned Sentence Transformer embeddings for semantic relevance.

- **Query Expansion**:
  - Suggests related legal terms to enhance retrieval scope.

---

## 4. Notes
Since it is kinda time-consuming for me to solo-coding this, while current time is peak season for my company's projects. I decided to cancel the competition, but hopefully you guys can see something useful from the pipeline i created. 

Here i added bm25s which convinced to be faster than the origin bm25, i did train the SupSimCSE-VoVanPhuc for like 2 or 3 days but it turned out, they did not allow to use. However, you can do ensemble with sbert and alibaba sentence transformers with adding in vocabs. 

I dont try to use the Generation part in RAG for the fact that i dont have a good feeling that it is a good option for me.

More things that can be further added that using vllm to create a second chance question for those exist more than one each query. I also highlighted the intention of the question so that the answer can be more specific during the search.

And yeap, you can give me a like or stars iusecase. Even raising an issue, if any in my code seemed confusing. Thank you boys <3 
