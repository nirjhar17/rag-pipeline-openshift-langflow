# Blog 1: Exploring Langflow — Building a RAG Pipeline on OpenShift

## Overview

This blog demonstrates how to build a complete **Retrieval-Augmented Generation (RAG)** pipeline visually using **Langflow** on **Red Hat OpenShift**. The pipeline uses:

- **Docling Serve** — PDF extraction and processing
- **LlamaStack** — Provides Granite embedding model and Qwen3 LLM
- **Milvus** — Vector database for storing and searching embeddings
- **Langflow** — Visual workflow orchestrator that ties everything together

### Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌────────────┐     ┌─────────┐
│  PDF (CAMS)  │────▶│  Docling Serve   │────▶│Export Docling│────▶│Split Text│
└─────────────┘     │  (CPU, OpenShift) │     │  Document   │     │ Chunks  │
                    └──────────────────┘     └────────────┘     └────┬────┘
                                                                      │
                    ┌──────────────────┐                              │
                    │   LlamaStack     │                              ▼
                    │ granite-embedding │─────────────────────▶┌──────────┐
                    │   -125m          │    embeddings         │  Milvus  │
                    └──────────────────┘                       │ (vector  │
                                                               │   DB)   │
                    ┌──────────────────┐                       └────┬────┘
                    │   LlamaStack     │                            │
                    │ Qwen3-0.6B (LLM) │◀──── retrieved context ───┘
                    └──────────────────┘
```

### OpenShift Cluster Details

| Component | Namespace | Endpoint |
|-----------|-----------|----------|
| Docling Serve | `rag-pipeline` | `http://docling-serve.rag-pipeline.svc.cluster.local:5001` |
| Langflow | `rag-pipeline` | `http://langflow-service.rag-pipeline.svc.cluster.local:7860` |
| Milvus | `rag-pipeline` | `http://milvus.rag-pipeline.svc.cluster.local:19530` |
| LlamaStack | `my-first-model` | `http://lsd-genai-playground-service.my-first-model.svc.cluster.local:8321` |
| Qwen3 LLM | `my-first-model` | `http://qwen3-0-6b-kserve-workload-svc.my-first-model.svc.cluster.local:8000` |

---

## Prerequisites

- OpenShift cluster with cluster-admin access
- Helm CLI installed
- `oc` CLI installed and logged in
- LlamaStack already deployed (with Granite embedding model and Qwen3 LLM)

```bash
oc login -u cluster-admin -p <password> https://api.<cluster>:443
```

---

## Step 1: Create the Namespace

```bash
oc new-project rag-pipeline
```

---

## Step 2: Deploy Docling Serve

Docling Serve converts PDFs into structured text/markdown. We use an init container to pre-download ML models into a PVC.

### 2.1 Create the PVC for model cache

```bash
oc apply -f manifests/docling-pvc.yaml
```

### 2.2 Deploy Docling Serve

```bash
oc apply -f manifests/docling-serve.yaml
```

This creates:
- A **Deployment** with an init container that downloads layout + tableformer models
- A **Service** on port 5001
- A **Route** with edge TLS termination

### 2.3 Verify

```bash
# Wait for init container to download models (~2-5 min)
oc get pods -n rag-pipeline -w

# Test the health endpoint
curl -k https://docling-serve-rag-pipeline.apps.<cluster>/health
# Expected: {"status":"ok"}
```

---

## Step 3: Deploy Langflow via Helm

### 3.1 Install Langflow

```bash
helm repo add langflow https://langflow-ai.github.io/langflow-helm-charts
helm repo update

helm install langflow-ide langflow/langflow-ide \
  --namespace rag-pipeline
```

### 3.2 Configure Langflow for OpenShift

```bash
# Enable auto-login (no API key required)
oc set env statefulset/langflow-service -n rag-pipeline \
  LANGFLOW_AUTO_LOGIN=true \
  LANGFLOW_SKIP_AUTH_AUTO_LOGIN=true

# Increase file upload size limit (for PDF uploads)
oc set env deployment/langflow-service-frontend -n rag-pipeline \
  LANGFLOW_MAX_FILE_SIZE_UPLOAD=100
```

### 3.3 Fix the Route

The default route needs timeout and upload size annotations:

```bash
oc apply -f manifests/langflow-route.yaml
```

Or patch the existing route:

```bash
oc annotate route langflow-ide -n rag-pipeline \
  haproxy.router.openshift.io/proxy-body-size=100m \
  haproxy.router.openshift.io/timeout=600s \
  --overwrite
```

### 3.4 Verify

Open in browser: `https://langflow-ide-rag-pipeline.apps.<cluster>`

---

## Step 4: Deploy Milvus (Vector Database)

### 4.1 Why standalone mode?

Milvus cluster mode deploys 10+ pods (Pulsar, datanode, mixcoord, querynode, etc.). For a blog demo, standalone mode is sufficient — just 3 pods: etcd, minio, and the standalone server.

### 4.2 Install Milvus

```bash
chmod +x manifests/milvus-install.sh
./manifests/milvus-install.sh
```

Or run the commands manually:

```bash
# Add Helm repo
helm repo add zilliztech https://zilliztech.github.io/milvus-helm/
helm repo update

# Grant anyuid SCC (OpenShift requires this for etcd/minio)
oc adm policy add-scc-to-user anyuid -z default -n rag-pipeline
oc adm policy add-scc-to-user anyuid -z milvus-etcd -n rag-pipeline
oc adm policy add-scc-to-user anyuid -z milvus-minio -n rag-pipeline

# Install standalone Milvus
helm install milvus zilliztech/milvus \
  --namespace rag-pipeline \
  --set cluster.enabled=false \
  --set standalone.resources.requests.cpu=500m \
  --set standalone.resources.requests.memory=2Gi \
  --set standalone.resources.limits.cpu=2 \
  --set standalone.resources.limits.memory=4Gi \
  --set streaming.enabled=false \
  --set woodpecker.enabled=false \
  --set pulsarv3.enabled=false \
  --set pulsar.enabled=false \
  --set etcd.replicaCount=1 \
  --set minio.mode=standalone \
  --set minio.resources.requests.memory=512Mi \
  --set etcd.resources.requests.memory=512Mi
```

### 4.3 Key Gotcha: OpenShift SCC

The etcd container runs as UID `1001`, which is outside OpenShift's default allowed range. Without the `anyuid` SCC grant, the etcd pod will be stuck in `FailedCreate` with:

```
pods "milvus-etcd-0" is forbidden: unable to validate against any security context constraint
```

### 4.4 Verify

```bash
# Wait ~90 seconds for readiness probe
oc get pods -n rag-pipeline | grep milvus

# Expected:
# milvus-etcd-0          1/1  Running
# milvus-minio-xxx       1/1  Running
# milvus-standalone-xxx  1/1  Running

# Test Milvus API from Langflow pod
oc exec langflow-service-0 -n rag-pipeline -- \
  curl -s http://milvus.rag-pipeline.svc.cluster.local:19530/v2/vectordb/collections/list \
  -X POST -H "Content-Type: application/json" -d '{}'
# Expected: {"code":0,"data":[]}
```

---

## Step 5: Verify LlamaStack Connectivity

LlamaStack is already running in the `my-first-model` namespace. Verify Langflow can reach it:

```bash
# Check available models
oc exec langflow-service-0 -n rag-pipeline -- \
  curl -s http://lsd-genai-playground-service.my-first-model.svc.cluster.local:8321/v1/models

# Expected: granite-embedding-125m, Qwen/Qwen3-0.6B
```

---

## Step 6: Build the Langflow Flow

Open the Langflow UI and use the **Vector Store RAG** template as a starting point. Then customize:

### 6.1 Ingestion Flow (Load Data)

```
Docling Serve → Export DoclingDocument → Split Text → Milvus
                                                       ↑
                          OpenAI Embeddings ───────────┘
                     (granite-embedding-125m via LlamaStack)
```

| Component | Configuration |
|-----------|---------------|
| **Docling Serve** | Server address: `http://docling-serve.rag-pipeline.svc.cluster.local:5001` |
| **Export DoclingDocument** | Export format: `Markdown`, Image export mode: `placeholder` |
| **Split Text** | Chunk Overlap: `200`, Chunk Size: `1000` |
| **OpenAI Embeddings** (from Bundles > OpenAI) | Model: `granite-embedding-125m`*, OpenAI API Base: `http://lsd-genai-playground-service.my-first-model.svc.cluster.local:8321/v1`, OpenAI API Key: `fake`, TikToken Enable: `false` |
| **Milvus** | Collection Name: `cams_docs_v2`, Connection URI: `http://milvus.rag-pipeline.svc.cluster.local:19530`, Primary Field: `pk`, Text Field: `text`, Vector Field: `vector` |

> **Important — Component Choice**: Use the **OpenAI Embeddings** component from the **OpenAI bundle** (Bundles > OpenAI), NOT the core "Embedding Model" component and NOT "HuggingFace Embeddings Inference". The OpenAI Embeddings bundle component has the `TikToken Enable` toggle and `OpenAI API Base` field exposed in its UI, making it the best fit for custom OpenAI-compatible endpoints like LlamaStack.

### 6.2 Retriever Flow (Search + LLM)

```
Chat Input → OpenAI Embeddings → Milvus (search) → Parser (Stringify)
                                                        │
Chat Input → Prompt ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘
                │
                ▼
         OpenAI Model (Qwen3) → Chat Output
```

| Component | Configuration |
|-----------|---------------|
| **Chat Input** | Input Text: your question (e.g., "What is the durability score of CAMS?") |
| **OpenAI Embeddings** (search) | Same config as ingestion: Model: `granite-embedding-125m`, OpenAI API Base: LlamaStack `/v1`, API Key: `fake`, TikToken Enable: `false` |
| **Milvus** (search) | Same collection `cams_docs_v2` and connection URI as ingestion Milvus |
| **Parser** | Mode: **Stringify** (converts Milvus search results list to plain text) |
| **Prompt** | Template: `{context}\n---\nGiven the context above, answer the question as best as possible.\nQuestion: {question}\nAnswer:` |
| **OpenAI Model** (LLM) | Model: `vllm-inference-1/Qwen/Qwen3-0.6B`*, OpenAI API Base: `http://lsd-genai-playground-service.my-first-model.svc.cluster.local:8321/v1`, API Key: `fake` |
| **Chat Output** | Displays the LLM response in the Playground |

> **Parser Mode**: Use **Stringify**, not Parser mode. Milvus returns a list of Data objects, and Parser mode throws `"List of Data objects is not supported"`. Stringify handles lists correctly.

> **Qwen3 Thinking Mode**: Qwen3 outputs `<think>...</think>` reasoning tags before the actual answer. This is a built-in feature of Qwen3 that cannot be easily disabled through LlamaStack. The answer appears after the `</think>` tag.

> *Note: The OpenAI Embeddings and OpenAI Model components both have hardcoded `DropdownInput` for model names. See Step 6.3 and 6.5 for the required code edits.

### 6.3 Component Code Fixes (Required for Both OpenAI Embeddings Components)

The **OpenAI Embeddings** bundle component (from Bundles > OpenAI) requires **two code edits** to work with LlamaStack's Granite embeddings. Click the `</>` code editor on each OpenAI Embeddings component.

#### Why Not Use Other Embedding Components?

We tried three embedding components before finding the right one:

| Component | Problem |
|-----------|---------|
| **Embedding Model** (core) | Hardcoded dropdown for model names, no `TikToken Enable` toggle, no `check_embedding_ctx_length` parameter |
| **HuggingFace Embeddings Inference** | [Known bug](https://github.com/langflow-ai/langflow/issues/6345) — concatenates endpoint URL with model name, creating invalid URL like `http://...8321/v1granite-embedding-125m` |
| **OpenAI Embeddings** (bundle) | Best fit — has `TikToken Enable` toggle and `OpenAI API Base` in advanced settings. Only needs two code edits (below) |

#### Fix 1: Custom Model Name (~line 40)

The model field is a locked `DropdownInput` with only OpenAI model names. Change it to free-text so you can type `granite-embedding-125m`:

```python
# FROM:
        DropdownInput(
            name="model",
            display_name="Model",
            advanced=False,
            options=OPENAI_EMBEDDING_MODEL_NAMES,
            value="text-embedding-3-small",
        ),

# TO:
        MessageTextInput(
            name="model",
            display_name="Model",
            advanced=False,
            value="granite-embedding-125m",
        ),
```

#### Fix 2: Disable Context Length Check (~line 75, inside `build_embeddings` method)

Even with TikToken disabled via the UI toggle, LangChain falls back to loading a HuggingFace tokenizer to check embedding context length. Since `granite-embedding-125m` isn't a valid HuggingFace model ID, this fails with `"granite-embedding-125m is not a local folder and is not a valid model identifier"`. Adding `check_embedding_ctx_length=False` skips this unnecessary client-side check (the Granite model on LlamaStack handles tokenization server-side):

```python
# Find this line in the build_embeddings method:
            tiktoken_enabled=self.tiktoken_enable,

# Add check_embedding_ctx_length=False right after it:
            tiktoken_enabled=self.tiktoken_enable,
            check_embedding_ctx_length=False,
```

#### UI Settings (Advanced)

After saving the code, also set these in the component's advanced settings:

- **TikToken Enable**: `false` (toggle off)
- **OpenAI API Base**: `http://lsd-genai-playground-service.my-first-model.svc.cluster.local:8321/v1`
- **OpenAI API Key**: `fake`

> **Important**: Both code fixes must be applied to EACH OpenAI Embeddings component independently. Editing one does not affect the other.

### 6.4 Verify Ingestion

After running the ingestion flow, verify data was stored in Milvus:

```bash
# Count documents
oc exec langflow-service-0 -n rag-pipeline -- \
  curl -s http://milvus.rag-pipeline.svc.cluster.local:19530/v2/vectordb/entities/query \
  -X POST -H "Content-Type: application/json" \
  -d '{"collectionName":"cams_docs_v2","filter":"pk > 0","limit":100,"outputFields":["pk","text"]}'

# View embeddings (768-dimensional vectors from Granite)
oc exec langflow-service-0 -n rag-pipeline -- \
  curl -s http://milvus.rag-pipeline.svc.cluster.local:19530/v2/vectordb/entities/query \
  -X POST -H "Content-Type: application/json" \
  -d '{"collectionName":"cams_docs_v2","filter":"pk > 0","limit":3,"outputFields":["pk","text","vector"]}'
```

Each row in Milvus stores: `pk` (unique ID), `text` (the original text chunk), and `vector` (768 floats from Granite embedding).

### 6.5 Code Fix for OpenAI Model (LLM) Component

The **OpenAI** bundle component (Bundles > OpenAI) for the LLM also has a locked dropdown. Click `</>` on the OpenAI (LLM) component and make two changes:

#### Fix 1: Custom Model Name (~line 48)

```python
# FROM:
        DropdownInput(
            name="model_name",
            display_name="Model Name",
            advanced=False,
            options=OPENAI_CHAT_MODEL_NAMES + OPENAI_REASONING_MODEL_NAMES,
            value=OPENAI_CHAT_MODEL_NAMES[0],
            combobox=True,
            real_time_refresh=True,
        ),

# TO:
        StrInput(
            name="model_name",
            display_name="Model Name",
            advanced=False,
            value="vllm-inference-1/Qwen/Qwen3-0.6B",
        ),
```

#### Fix 2: Remove Leaked Display Name (~line 80, inside `build_model` method)

Langflow leaks the `display_name` ("Model Name") into `model_kwargs`, causing `AsyncCompletions.create() got an unexpected keyword argument 'Model Name'`. Add one line to strip it:

```python
# Find this line:
        model_kwargs = self.model_kwargs or {}

# Add right after it:
        model_kwargs = self.model_kwargs or {}
        model_kwargs.pop("Model Name", None)
```

#### UI Settings (Advanced)

After saving the code:

- **OpenAI API Base**: `http://lsd-genai-playground-service.my-first-model.svc.cluster.local:8321/v1`
- **OpenAI API Key**: `fake`

### 6.6 Test in Playground

Open the **Playground** (bottom-right corner of the Langflow UI) and ask questions about your CAMS document:

- "What is the durability score of CAMS?"
- "What are the key financial metrics?"
- "What is this document about?"

The full retriever flow runs:
1. Your question is embedded by Granite (same model that embedded the chunks)
2. Milvus finds the most similar chunks by comparing vectors
3. Parser converts search results to text
4. Prompt combines the context + your question
5. Qwen3 reads the context and generates an answer

> **Note**: Qwen3 includes `<think>...</think>` reasoning tags in responses. The actual answer appears after `</think>`. This is a model feature that cannot be disabled via LlamaStack's API proxy.

---

## Troubleshooting

### Error: `413 Request Entity Too Large`
**Cause**: OpenShift route or Langflow's Nginx rejects large PDF uploads.
**Fix**:
```bash
oc annotate route langflow-ide -n rag-pipeline \
  haproxy.router.openshift.io/proxy-body-size=100m --overwrite
oc set env deployment/langflow-service-frontend -n rag-pipeline \
  LANGFLOW_MAX_FILE_SIZE_UPLOAD=100
```

### Error: `Flow build failed - Network error` after ~30s
**Cause**: OpenShift route default timeout is 30 seconds.
**Fix**:
```bash
oc annotate route langflow-ide -n rag-pipeline \
  haproxy.router.openshift.io/timeout=600s --overwrite
```

### Error: `Invalid collection name: cams-docs`
**Cause**: Milvus collection names cannot contain hyphens.
**Fix**: Use underscores: `cams_docs`

### Error: `uri: ttp://... is illegal`
**Cause**: Missing `h` in `http://` in the Milvus Connection URI.
**Fix**: Ensure the URI starts with `http://`

### Error: `Embeddings.create() got an unexpected keyword argument 'Model Name'`
**Cause**: The core Embedding Model component passes `display_name` as kwarg during Milvus search.
**Fix**: Edit the component code to change `DropdownInput` to `MessageTextInput` for the model field (see Step 6.3).

### Error: `unrecognized dtype for key: doc`
**Cause**: Docling Serve outputs a `DoclingDocument` object, not plain text. Milvus can't store the complex `doc` field.
**Fix**: Add an **Export DoclingDocument** component between Docling Serve and Split Text.

### Error: `Text key 'text' not found in DataFrame columns`
**Cause**: Split Text expects a `text` column but receives a `DoclingDocument` column.
**Fix**: Same as above — add Export DoclingDocument to convert to Markdown first.

### Error: `No vector field is found`
**Cause**: The Embedding output is not connected to the Milvus Embedding port, OR the HuggingFace Embeddings Inference component has a known bug that silently fails.
**Fix**: Use the core Embedding Model component (with OpenAI provider) instead of HuggingFace Embeddings Inference. Connect Embedding Model → Embeddings output to Milvus → Embedding input port.

### Error: `Input should be a valid string` (400 from LlamaStack)
**Cause**: LangChain's `OpenAIEmbeddings` uses tiktoken to pre-tokenize text into integer token IDs. LlamaStack only accepts plain strings.
**Fix**: Toggle **TikToken Enable** to `false` in the OpenAI Embeddings component's advanced settings. Also add `check_embedding_ctx_length=False` in the component code (see Step 6.3, Fix 2).

### Error: `granite-embedding-125m is not a local folder and is not a valid model identifier`
**Cause**: With TikToken disabled, LangChain falls back to loading a HuggingFace tokenizer (`AutoTokenizer.from_pretrained("granite-embedding-125m")`) to check embedding context length. `granite-embedding-125m` isn't a valid HuggingFace repo ID (the full ID would be `ibm-granite/granite-embedding-125m-english`).
**Fix**: Add `check_embedding_ctx_length=False` to the `OpenAIEmbeddings()` constructor in the component code. This skips the unnecessary client-side tokenizer — the Granite model on LlamaStack handles tokenization server-side.

### Error: Milvus etcd pod stuck in `FailedCreate`
**Cause**: OpenShift SCC blocks etcd from running as UID 1001.
**Fix**:
```bash
oc adm policy add-scc-to-user anyuid -z default -n rag-pipeline
oc adm policy add-scc-to-user anyuid -z milvus-etcd -n rag-pipeline
oc adm policy add-scc-to-user anyuid -z milvus-minio -n rag-pipeline
```

### Error: `AsyncCompletions.create() got an unexpected keyword argument 'Model Name'`
**Cause**: Langflow leaks the `display_name` ("Model Name") of the OpenAI component's model field into `model_kwargs`, which gets passed to the OpenAI client's `create()` method.
**Fix**: Edit the OpenAI component code and add `model_kwargs.pop("Model Name", None)` after the line `model_kwargs = self.model_kwargs or {}` (see Step 6.5, Fix 2).

### Error: `List of Data objects is not supported` (Parser component)
**Cause**: The Parser component is in **Parser** mode, which only handles a single Data object. Milvus returns a list of search results.
**Fix**: Switch the Parser to **Stringify** mode.

### Error: Milvus deployed in cluster mode (10+ pods)
**Cause**: Default Helm values enable cluster mode with Pulsar.
**Fix**: Use `--set cluster.enabled=false --set pulsarv3.enabled=false --set streaming.enabled=false`

---

## Current Status

| Component | Status |
|-----------|--------|
| Docling Serve | Running |
| Langflow | Running |
| Milvus (standalone) | Running |
| LlamaStack (Granite + Qwen3) | Running |
| **Ingestion flow** | **Complete** — 34 CAMS PDF chunks stored in Milvus with 768-dim Granite embeddings |
| **Retriever flow** | **Complete** — End-to-end RAG working in Playground |

### Ingestion Results

```
Collection: cams_docs_v2
Documents:  34 chunks
Vector dim: 768 (Granite embedding-125m)
Fields:     pk (Int64), text (VarChar), vector (FloatVector)
```

### Sample RAG Query

**Question**: "What are the prices of CAMS?"

**Response**: Qwen3 retrieves relevant financial context from Milvus and answers:
> The current P/E ratio for Computer Age Management Services Ltd. is 40.5, based on the P/E Buy Sell Zone analysis.

(Qwen3 includes `<think>` reasoning tags before the answer — this is a model feature.)

---

## Summary of All Code Edits

All three Langflow components with hardcoded dropdowns needed code fixes via the `</>` editor:

| Component | Fix | What Changed |
|-----------|-----|-------------|
| **OpenAI Embeddings** (x2) | Fix 1: `DropdownInput` → `MessageTextInput` for model | Allows typing `granite-embedding-125m` |
| **OpenAI Embeddings** (x2) | Fix 2: `check_embedding_ctx_length=False` | Stops HuggingFace tokenizer fallback |
| **OpenAI Model** (x1) | Fix 1: `DropdownInput` → `StrInput` for model_name | Allows typing `vllm-inference-1/Qwen/Qwen3-0.6B` |
| **OpenAI Model** (x1) | Fix 2: `model_kwargs.pop("Model Name", None)` | Removes leaked display name from API call |

---

## Next Steps

1. Take screenshots for the blog
2. (Blog 2) Evaluate RAG quality using TrustyAI on OpenShift
