# Exploring Langflow: Building a RAG Pipeline on OpenShift

A hands-on guide to visually building a Retrieval-Augmented Generation pipeline using Langflow, Docling, Milvus, and LlamaStack — all running on Red Hat OpenShift.

Part 1 of 2

---

## What We Are Building

In this blog, we will build a complete RAG pipeline — entirely visually — using Langflow on a Red Hat OpenShift cluster. The pipeline takes a PDF document, extracts text from it, splits it into chunks, embeds each chunk into a vector, stores those vectors in a database, and then lets you ask questions about the document using an LLM.

Here is the stack:

- Docling Serve — converts PDFs into structured Markdown text. Deployed on OpenShift using a YAML manifest.
- Langflow — a low-code visual AI workflow builder. Deployed on OpenShift via Helm chart.
- Milvus — a vector database that stores embeddings and performs similarity search. Deployed on OpenShift via Helm chart in standalone mode.
- LlamaStack — Red Hat's unified AI runtime, providing the Granite embedding-125m model for embeddings and Qwen3-0.6B as the LLM. Already deployed on the cluster via OpenShift AI.

Everything runs inside the cluster. There are no external API calls to OpenAI or any other cloud service. The embeddings and LLM inference happen on self-hosted models via LlamaStack, which exposes an OpenAI-compatible API.

## Architecture

```
PDF (CAMS) ──▶ Docling Serve ──▶ Export DoclingDocument ──▶ Split Text
                                                                │
                                                                ▼
LlamaStack (granite-embedding-125m) ──── embeddings ──▶ Milvus (vector DB)
                                                                │
LlamaStack (Qwen3-0.6B LLM) ◀──── retrieved context ──────────┘
```

The pipeline has two flows inside Langflow. The ingestion flow processes the PDF and stores chunks in Milvus. The retriever flow takes a user question, searches Milvus for relevant chunks, and passes them to Qwen3 to generate an answer.

## OpenShift Cluster Details

All components run in two namespaces on the cluster:

- Docling Serve — namespace `rag-pipeline`, endpoint `http://docling-serve.rag-pipeline.svc.cluster.local:5001`
- Langflow — namespace `rag-pipeline`, endpoint `http://langflow-service.rag-pipeline.svc.cluster.local:7860`
- Milvus — namespace `rag-pipeline`, endpoint `http://milvus.rag-pipeline.svc.cluster.local:19530`
- LlamaStack — namespace `my-first-model`, endpoint `http://lsd-genai-playground-service.my-first-model.svc.cluster.local:8321`
- Qwen3 LLM — namespace `my-first-model`, endpoint `http://qwen3-0-6b-kserve-workload-svc.my-first-model.svc.cluster.local:8000`

## Prerequisites

Before starting, make sure you have:

- An OpenShift cluster with cluster-admin access
- The Helm CLI installed
- The `oc` CLI installed and logged in
- LlamaStack already deployed with the Granite embedding model and Qwen3 LLM

```
oc login -u cluster-admin -p <password> https://api.<cluster>:443
```

---

## Step 1: Create the Namespace

Start by creating a dedicated namespace for the RAG pipeline components.

```
oc new-project rag-pipeline
```

---

## Step 2: Deploy Docling Serve

Docling Serve converts PDFs into structured text and Markdown. We deploy it using a YAML manifest that includes a Deployment, a Service, and a Route. The Deployment uses an init container to pre-download the layout and tableformer ML models into a PersistentVolumeClaim so they are cached across pod restarts.

### 2.1 Create the PVC for model cache

```
oc apply -f manifests/docling-pvc.yaml
```

### 2.2 Deploy Docling Serve

```
oc apply -f manifests/docling-serve.yaml
```

This creates a Deployment with an init container that downloads the models, a ClusterIP Service on port 5001, and a Route with edge TLS termination.

### 2.3 Verify

Wait for the init container to finish downloading models (about 2 to 5 minutes), then verify the health endpoint.

```
# Watch pods until docling-serve shows 1/1 Running
oc get pods -n rag-pipeline -w

# Test the health endpoint
curl -k https://docling-serve-rag-pipeline.apps.<cluster>/health
# Expected: {"status":"ok"}
```

---

## Step 3: Deploy Langflow via Helm

Langflow is the visual workflow builder where we design and run our RAG flows. We install it using the official Langflow Helm chart.

### 3.1 Install Langflow

```
helm repo add langflow https://langflow-ai.github.io/langflow-helm-charts
helm repo update

helm install langflow-ide langflow/langflow-ide \
  --namespace rag-pipeline
```

### 3.2 Configure Langflow for OpenShift

By default, Langflow requires login credentials. For a demo environment, we enable auto-login and increase the file upload size limit so we can upload large PDFs.

```
# Enable auto-login (no API key required)
oc set env statefulset/langflow-service -n rag-pipeline \
  LANGFLOW_AUTO_LOGIN=true \
  LANGFLOW_SKIP_AUTH_AUTO_LOGIN=true

# Increase file upload size limit for PDF uploads
oc set env deployment/langflow-service-frontend -n rag-pipeline \
  LANGFLOW_MAX_FILE_SIZE_UPLOAD=100
```

### 3.3 Fix the Route

The default OpenShift route has a 30-second timeout, which is too short for Langflow flows that involve PDF processing and LLM inference. We also need to increase the upload body size. Apply the route manifest or patch the existing route.

```
oc apply -f manifests/langflow-route.yaml
```

Alternatively, you can patch the existing route directly:

```
oc annotate route langflow-ide -n rag-pipeline \
  haproxy.router.openshift.io/proxy-body-size=100m \
  haproxy.router.openshift.io/timeout=600s \
  --overwrite
```

### 3.4 Verify

Open the Langflow UI in your browser at `https://langflow-ide-rag-pipeline.apps.<cluster>`. You should see the Langflow dashboard without any login prompt.

---

## Step 4: Deploy Milvus (Vector Database)

### Why standalone mode?

Milvus cluster mode deploys 10+ pods including Pulsar, datanode, mixcoord, and querynode. For a blog demo, standalone mode is sufficient and much simpler. It only needs 3 pods: etcd, minio, and the standalone Milvus server.

### 4.1 Install Milvus via Helm

We provide a shell script that handles the full installation, including the OpenShift-specific security context grants.

```
chmod +x manifests/milvus-install.sh
./manifests/milvus-install.sh
```

If you prefer to run the commands manually, here is what the script does:

```
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

### 4.2 Key Gotcha: OpenShift SCC

The etcd container runs as UID 1001, which is outside OpenShift's default allowed UID range. Without the anyuid SCC grant, the etcd pod will be stuck in FailedCreate with the error:

```
pods "milvus-etcd-0" is forbidden: unable to validate against any security context constraint
```

### 4.3 Verify

Wait about 90 seconds for the readiness probes, then check the pods and test the Milvus API.

```
# Check Milvus pods
oc get pods -n rag-pipeline | grep milvus

# Expected:
# milvus-etcd-0          1/1  Running
# milvus-minio-xxx       1/1  Running
# milvus-standalone-xxx  1/1  Running

# Test Milvus API from the Langflow pod
oc exec langflow-service-0 -n rag-pipeline -- \
  curl -s http://milvus.rag-pipeline.svc.cluster.local:19530/v2/vectordb/collections/list \
  -X POST -H "Content-Type: application/json" -d '{}'
# Expected: {"code":0,"data":[]}
```

---

## Step 5: Verify LlamaStack Connectivity

LlamaStack is already running in the `my-first-model` namespace. Before building the Langflow flow, verify that Langflow can reach the LlamaStack API.

```
oc exec langflow-service-0 -n rag-pipeline -- \
  curl -s http://lsd-genai-playground-service.my-first-model.svc.cluster.local:8321/v1/models

# Expected output should list: granite-embedding-125m, Qwen/Qwen3-0.6B
```

---

## Step 6: Build the Langflow Flow

Open the Langflow UI and use the Vector Store RAG template as a starting point. We will customize it with our OpenShift-hosted components.

### 6.1 Ingestion Flow (Load Data)

The ingestion flow processes the PDF and stores the resulting chunks in Milvus. The flow is:

```
Docling Serve ──▶ Export DoclingDocument ──▶ Split Text ──▶ Milvus
                                                             ↑
                        OpenAI Embeddings ───────────────────┘
                   (granite-embedding-125m via LlamaStack)
```

Here is how to configure each component:

Docling Serve — Set the server address to `http://docling-serve.rag-pipeline.svc.cluster.local:5001` and upload your PDF file.

Export DoclingDocument — Set the export format to Markdown and the image export mode to placeholder. This component is necessary because Docling outputs a DoclingDocument object, and downstream components need plain text.

Split Text — Set the chunk overlap to 200 and the chunk size to 1000. Leave the separator empty for the default.

OpenAI Embeddings (from Bundles, then OpenAI) — Set the model to `granite-embedding-125m`, the OpenAI API Base to `http://lsd-genai-playground-service.my-first-model.svc.cluster.local:8321/v1`, the API Key to `fake`, and toggle TikToken Enable to false. This component requires code edits — see section 6.3 below.

Milvus — Set the collection name to `cams_docs_v2`, the connection URI to `http://milvus.rag-pipeline.svc.cluster.local:19530`, the primary field to `pk`, the text field to `text`, and the vector field to `vector`.

Important — Component Choice: Use the OpenAI Embeddings component from the OpenAI bundle (Bundles, then OpenAI), NOT the core "Embedding Model" component and NOT "HuggingFace Embeddings Inference". The OpenAI Embeddings bundle component has the TikToken Enable toggle and OpenAI API Base field exposed in its UI, making it the best fit for custom OpenAI-compatible endpoints like LlamaStack.

### 6.2 Retriever Flow (Search + LLM)

The retriever flow takes a user question, searches Milvus for relevant chunks, and passes them to the LLM. The flow is:

```
Chat Input ──▶ OpenAI Embeddings ──▶ Milvus (search) ──▶ Parser (Stringify)
                                                               │
Chat Input ──▶ Prompt ◀──── context ───────────────────────────┘
                 │
                 ▼
          OpenAI Model (Qwen3) ──▶ Chat Output
```

Here is how to configure each component:

Chat Input — This is where you type your question in the Playground, for example "What is the durability score of CAMS?"

OpenAI Embeddings (search) — Same configuration as the ingestion embeddings: model `granite-embedding-125m`, API Base pointing to LlamaStack, API Key `fake`, TikToken disabled. The same code edits are required.

Milvus (search) — Same collection name `cams_docs_v2` and connection URI as the ingestion Milvus.

Parser — Set the mode to Stringify. This is critical. Milvus returns a list of Data objects, and the default Parser mode only handles a single Data object. It will throw "List of Data objects is not supported" if you use Parser mode. Stringify correctly serializes the entire list into plain text.

Prompt — Use the following template:

```
{context}
---
Given the context above, answer the question as best as possible.
Question: {question}
Answer:
```

Connect the Parser output to the `{context}` variable and the Chat Input to the `{question}` variable.

OpenAI Model (LLM) — Set the model to `vllm-inference-1/Qwen/Qwen3-0.6B`, the OpenAI API Base to `http://lsd-genai-playground-service.my-first-model.svc.cluster.local:8321/v1`, and the API Key to `fake`. This component also requires code edits — see section 6.5 below.

Chat Output — Displays the final LLM response in the Playground.

Note on Qwen3 Thinking Mode: Qwen3 outputs `<think>...</think>` reasoning tags before the actual answer. This is a built-in feature of Qwen3 that cannot be easily disabled through LlamaStack. The actual answer appears after the `</think>` tag.

### 6.3 Code Fixes for OpenAI Embeddings Components

The OpenAI Embeddings bundle component requires two code edits to work with LlamaStack's Granite embeddings. Click the `</>` code editor button on each OpenAI Embeddings component and make these changes.

Why not use other embedding components?

We tried three embedding components before finding the right one:

- Embedding Model (core) — has a hardcoded dropdown for model names, no TikToken Enable toggle, and no check_embedding_ctx_length parameter. Required extensive code hacks.
- HuggingFace Embeddings Inference — has a known bug (github.com/langflow-ai/langflow/issues/6345) that concatenates the endpoint URL with the model name, creating an invalid URL like `http://...8321/v1granite-embedding-125m`.
- OpenAI Embeddings (bundle) — best fit. Has the TikToken Enable toggle and OpenAI API Base field in advanced settings. Only needs two small code edits.

Fix 1: Custom Model Name (around line 40)

The model field is a locked DropdownInput that only shows OpenAI model names. Change it to a free-text input so you can type `granite-embedding-125m`.

```
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

Fix 2: Disable Context Length Check (around line 75)

Even with TikToken disabled via the UI toggle, LangChain falls back to loading a HuggingFace tokenizer to check embedding context length. Since `granite-embedding-125m` is not a valid HuggingFace model ID, this fails with the error "granite-embedding-125m is not a local folder and is not a valid model identifier". Adding `check_embedding_ctx_length=False` skips this unnecessary client-side check. The Granite model on LlamaStack handles tokenization server-side.

```
# Find this line in the build_embeddings method:
            tiktoken_enabled=self.tiktoken_enable,

# Add check_embedding_ctx_length=False right after it:
            tiktoken_enabled=self.tiktoken_enable,
            check_embedding_ctx_length=False,
```

After saving the code, also set these in the component's advanced settings: TikToken Enable to false, OpenAI API Base to the LlamaStack endpoint, and OpenAI API Key to `fake`.

Important: Both code fixes must be applied to EACH OpenAI Embeddings component independently. Editing one does not affect the other.

### 6.4 Verify Ingestion

After running the ingestion flow, verify that data was stored in Milvus by querying directly from the Langflow pod.

```
# Count documents in the collection
oc exec langflow-service-0 -n rag-pipeline -- \
  curl -s http://milvus.rag-pipeline.svc.cluster.local:19530/v2/vectordb/entities/query \
  -X POST -H "Content-Type: application/json" \
  -d '{"collectionName":"cams_docs_v2","filter":"pk > 0","limit":100,"outputFields":["pk","text"]}'
```

Each row in Milvus stores three fields: `pk` (a unique integer ID), `text` (the original text chunk), and `vector` (768 floats from the Granite embedding).

### 6.5 Code Fix for OpenAI Model (LLM) Component

The OpenAI bundle component for the LLM also has a locked dropdown for model names. Click `</>` on the OpenAI (LLM) component and make two changes.

Fix 1: Custom Model Name (around line 48)

```
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

Fix 2: Remove Leaked Display Name (around line 80)

Langflow has a bug where it leaks the display_name ("Model Name" with a space) into model_kwargs. This gets passed to the OpenAI client and causes the error: `AsyncCompletions.create() got an unexpected keyword argument 'Model Name'`. Add two lines to strip it.

```
# Find this line in the build_model method:
        model_kwargs = self.model_kwargs or {}

# Add right after it:
        model_kwargs = self.model_kwargs or {}
        model_kwargs.pop("Model Name", None)
        model_kwargs.pop("", None)
```

After saving the code, set the OpenAI API Base to the LlamaStack endpoint and the API Key to `fake` in the component's advanced settings.

### 6.6 Test in Playground

Open the Playground from the bottom-right corner of the Langflow UI and ask questions about your document:

- "What is the durability score of CAMS?"
- "What are the key financial metrics?"
- "What is this document about?"

The full retriever flow runs like this: your question is embedded by Granite (the same model that embedded the chunks), Milvus finds the most similar chunks by comparing vectors, the Parser converts the search results to text, the Prompt combines the context with your question, and Qwen3 reads the context and generates an answer.

---

## Troubleshooting

Here is a summary of every error we encountered during development, along with the fixes.

413 Request Entity Too Large — The OpenShift route or Langflow's Nginx rejects large PDF uploads. Fix it by annotating the route with `haproxy.router.openshift.io/proxy-body-size=100m` and setting the `LANGFLOW_MAX_FILE_SIZE_UPLOAD=100` environment variable.

Flow build failed — Network error after 30 seconds — The OpenShift route has a default timeout of 30 seconds. Fix it by annotating the route with `haproxy.router.openshift.io/timeout=600s`.

Invalid collection name: cams-docs — Milvus collection names cannot contain hyphens. Use underscores instead: `cams_docs_v2`.

uri: ttp://... is illegal — A typo in the Milvus Connection URI. Make sure the URI starts with `http://`.

unrecognized dtype for key: doc — Docling Serve outputs a DoclingDocument object, not plain text. Milvus cannot store the complex doc field. Fix it by adding an Export DoclingDocument component between Docling Serve and Split Text.

Text key 'text' not found in DataFrame columns — Split Text expects a text column but receives a DoclingDocument column. Same fix as above: add Export DoclingDocument to convert to Markdown first.

No vector field is found — The Embedding output is not connected to the Milvus Embedding port, or the HuggingFace Embeddings Inference component has a known bug that silently fails. Fix it by using the OpenAI Embeddings component and making sure the Embeddings output port is connected to Milvus's Embedding input port.

Input should be a valid string (400 from LlamaStack) — LangChain's OpenAIEmbeddings uses tiktoken to pre-tokenize text into integer token IDs. LlamaStack only accepts plain strings. Fix it by toggling TikToken Enable to false and adding `check_embedding_ctx_length=False` in the component code.

granite-embedding-125m is not a local folder and is not a valid model identifier — With TikToken disabled, LangChain falls back to loading a HuggingFace tokenizer. Since granite-embedding-125m is not a valid HuggingFace repo ID, it fails. Fix it by adding `check_embedding_ctx_length=False` to skip the unnecessary client-side tokenizer.

Milvus etcd pod stuck in FailedCreate — OpenShift SCC blocks etcd from running as UID 1001. Fix it by granting the anyuid SCC to the default, milvus-etcd, and milvus-minio service accounts.

AsyncCompletions.create() got an unexpected keyword argument 'Model Name' — Langflow leaks the display_name of the model field into model_kwargs. Fix it by adding `model_kwargs.pop("Model Name", None)` in the component code.

List of Data objects is not supported (Parser component) — The Parser component is in Parser mode, which only handles a single Data object. Milvus returns a list of search results. Fix it by switching the Parser to Stringify mode.

Milvus deployed in cluster mode (10+ pods) — The default Helm values enable cluster mode with Pulsar. Fix it by adding `--set cluster.enabled=false --set pulsarv3.enabled=false --set streaming.enabled=false` to the Helm install command.

---

## Results

After completing both the ingestion and retriever flows, the pipeline is fully operational.

The ingestion flow stored 34 chunks from the CAMS PDF into a Milvus collection called `cams_docs_v2`. Each chunk has a unique ID, the original text, and a 768-dimensional vector from the Granite embedding-125m model.

For a sample query like "What are the prices of CAMS?", Qwen3 retrieves the relevant financial context from Milvus and returns:

> The current P/E ratio for Computer Age Management Services Ltd. is 40.5, based on the P/E Buy Sell Zone analysis.

## Summary of All Code Edits

All three Langflow components with hardcoded dropdowns needed code fixes via the `</>` editor:

- OpenAI Embeddings (applied to both ingestion and search instances) — Fix 1: Changed DropdownInput to MessageTextInput for the model field, allowing us to type `granite-embedding-125m`. Fix 2: Added `check_embedding_ctx_length=False` to stop the HuggingFace tokenizer fallback.
- OpenAI Model — Fix 1: Changed DropdownInput to StrInput for model_name, allowing us to type `vllm-inference-1/Qwen/Qwen3-0.6B`. Fix 2: Added `model_kwargs.pop("Model Name", None)` to remove the leaked display name from the API call.

---

## Next Steps

In Part 2 of this series, we will evaluate the quality of our RAG pipeline using TrustyAI on OpenShift — measuring retrieval accuracy, answer relevance, and faithfulness.

The complete source code, manifests, and deployment scripts are available on GitHub: https://github.com/nirjhar17/rag-pipeline-openshift-langflow
