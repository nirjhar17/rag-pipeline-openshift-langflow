# RAG Pipeline on OpenShift with Langflow

A visual RAG (Retrieval-Augmented Generation) pipeline built on Red Hat OpenShift using Langflow as the workflow orchestrator. **End-to-end pipeline complete** — 34 CAMS PDF chunks stored in Milvus with 768-dim Granite embeddings, retriever flow querying Qwen3 via LlamaStack.

## Stack

| Component | Purpose | Image/Chart |
|-----------|---------|-------------|
| **Langflow** | Visual workflow builder | `langflow/langflow-ide` Helm chart |
| **Docling Serve** | PDF extraction → Markdown | `quay.io/docling-project/docling-serve-cpu:latest` |
| **Milvus** | Vector database (standalone) | `zilliztech/milvus` Helm chart |
| **LlamaStack** | Granite embedding-125m + Qwen3-0.6B LLM | Already deployed via OpenShift AI |

## Langflow Components Used

| Langflow Component | Purpose | Key Setting |
|-------------------|---------|-------------|
| **Docling Serve** | PDF → structured text | Server: internal Docling service |
| **Export DoclingDocument** | DoclingDocument → Markdown | Export format: Markdown |
| **Split Text** | Text → chunks | 1000 chars, 200 overlap |
| **OpenAI Embeddings** (Bundles > OpenAI) x2 | Chunks → 768-dim vectors | Model: `granite-embedding-125m` via LlamaStack |
| **Milvus** x2 | Vector storage + search | Collection: `cams_docs_v2` |
| **Parser** | Milvus results → text | Mode: Stringify |
| **Prompt** | Combine context + question | Template with `{context}` and `{question}` |
| **OpenAI Model** (Bundles > OpenAI) | LLM for chat | Model: `vllm-inference-1/Qwen/Qwen3-0.6B` via LlamaStack |
| **Chat Input / Chat Output** | Playground I/O | — |

## Quick Start

```bash
# 1. Create namespace
oc new-project rag-pipeline

# 2. Deploy Docling Serve
oc apply -f manifests/docling-pvc.yaml
oc apply -f manifests/docling-serve.yaml

# 3. Deploy Langflow
helm repo add langflow https://langflow-ai.github.io/langflow-helm-charts
helm install langflow-ide langflow/langflow-ide -n rag-pipeline
oc set env statefulset/langflow-service -n rag-pipeline LANGFLOW_AUTO_LOGIN=true LANGFLOW_SKIP_AUTH_AUTO_LOGIN=true
oc set env deployment/langflow-service-frontend -n rag-pipeline LANGFLOW_MAX_FILE_SIZE_UPLOAD=100
oc apply -f manifests/langflow-route.yaml

# 4. Deploy Milvus
chmod +x manifests/milvus-install.sh
./manifests/milvus-install.sh
```

## Internal Endpoints

```
Docling:    http://docling-serve.rag-pipeline.svc.cluster.local:5001
Langflow:   http://langflow-service.rag-pipeline.svc.cluster.local:7860
Milvus:     http://milvus.rag-pipeline.svc.cluster.local:19530
LlamaStack: http://lsd-genai-playground-service.my-first-model.svc.cluster.local:8321
```

## Blog

See [BLOG.md](BLOG.md) for the full step-by-step guide with screenshots and troubleshooting.

## Structure

```
rag-pipeline-blog/
├── README.md                 # This file
├── BLOG.md                   # Full blog post
├── LANGFLOW-FLOW.md          # Langflow flow configuration details
└── manifests/
    ├── docling-pvc.yaml      # PVC for Docling model cache
    ├── docling-serve.yaml    # Docling Serve Deployment + Service + Route
    ├── langflow-route.yaml   # Langflow Route with timeout/upload fixes
    └── milvus-install.sh     # Milvus standalone Helm install script
```
