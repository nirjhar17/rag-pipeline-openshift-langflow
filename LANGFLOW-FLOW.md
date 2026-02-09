# Langflow Flow Configuration

## Flow Overview

The RAG pipeline in Langflow has two halves:

1. **Ingestion Flow** (bottom) — Loads PDF, chunks text, generates embeddings, stores in Milvus
2. **Retriever Flow** (top) — Takes user query, searches Milvus, passes context to LLM

## Embedding Component Choice

We tested three embedding components before finding the right one:

| Component | Where to Find | Result |
|-----------|---------------|--------|
| **Embedding Model** (core) | Components > Models | Hardcoded model dropdown, no TikToken toggle, no `check_embedding_ctx_length` — requires extensive code hacks |
| **HuggingFace Embeddings Inference** | Bundles > HuggingFace | [Known bug](https://github.com/langflow-ai/langflow/issues/6345) — concatenates URL + model name, silently fails |
| **OpenAI Embeddings** (bundle) | **Bundles > OpenAI** | **Best fit** — has `TikToken Enable` toggle, `OpenAI API Base` field, only needs 2 small code edits |

**Use the OpenAI Embeddings component from Bundles > OpenAI** for both ingestion and search.

## Ingestion Flow Components

```
Docling Serve → Export DoclingDocument → Split Text → Milvus (Ingest)
                                                        ↑
                     OpenAI Embeddings ─────────────────┘
                (granite-embedding-125m via LlamaStack)
```

### Docling Serve
- **Files**: Upload your PDF (e.g., Cams.pdf)
- **Server address**: `http://docling-serve.rag-pipeline.svc.cluster.local:5001`

### Export DoclingDocument
- **Export format**: Markdown
- **Image export mode**: placeholder
- **Why needed**: Docling outputs a `DoclingDocument` object. This component converts it to plain Markdown text that Split Text can process.

### Split Text
- **Chunk Overlap**: 200
- **Chunk Size**: 1000
- **Separator**: (leave empty for default)

### OpenAI Embeddings (Ingestion)
- **Model**: `granite-embedding-125m` (requires code edit — see below)
- **OpenAI API Key**: `fake`
- **OpenAI API Base** (advanced): `http://lsd-genai-playground-service.my-first-model.svc.cluster.local:8321/v1`
- **TikToken Enable** (advanced): `false`

### Milvus (Ingestion)
- **Collection Name**: `cams_docs_v2` (no hyphens! Milvus only allows letters, numbers, underscores)
- **Connection URI**: `http://milvus.rag-pipeline.svc.cluster.local:19530`
- **Primary Field Name**: `pk`
- **Text Field Name**: `text`
- **Vector Field Name**: `vector`
- **Ingest Data**: Connected from Split Text → Chunks
- **Embedding**: Connected from OpenAI Embeddings → Embeddings

### Ingestion Results
After running the ingestion flow:
- **34 chunks** stored in Milvus collection `cams_docs_v2`
- Each row: `pk` (unique ID) + `text` (original chunk) + `vector` (768-dim Granite embedding)

## Retriever Flow Components

```
Chat Input ──→ OpenAI Embeddings (search) ──→ Milvus (search) ──→ Parser (Stringify)
                                                                        │
Chat Input ──→ Prompt ◀──── context ────────────────────────────────────┘
                  │
                  ▼
           OpenAI Model (Qwen3) ──→ Chat Output
```

### Chat Input
- Your question, typed in the Playground (e.g., "What is the durability score of CAMS?")

### OpenAI Embeddings (Search)
- Same configuration as the ingestion OpenAI Embeddings (same code edits required)
- **Model**: `granite-embedding-125m`
- **OpenAI API Key**: `fake`
- **OpenAI API Base** (advanced): `http://lsd-genai-playground-service.my-first-model.svc.cluster.local:8321/v1`
- **TikToken Enable** (advanced): `false`

### Milvus (Search)
- Same Collection Name (`cams_docs_v2`) and Connection URI as the ingestion Milvus
- **Search Query**: Connected from Chat Input

### Parser
- **Mode**: **Stringify** (critical — do NOT use "Parser" mode)
- **Why Stringify**: Milvus returns a *list* of Data objects. Parser mode only handles a single Data object and throws `"List of Data objects is not supported"`. Stringify mode correctly serializes the entire list into plain text.

### Prompt
- **Template**:
  ```
  {context}
  ---
  Given the context above, answer the question as best as possible.
  Question: {question}
  Answer:
  ```
- **Connections**: `{context}` ← Parser output, `{question}` ← Chat Input

### OpenAI Model (LLM)
- **Model Name**: `vllm-inference-1/Qwen/Qwen3-0.6B` (requires code edit — see below)
- **OpenAI API Base**: `http://lsd-genai-playground-service.my-first-model.svc.cluster.local:8321/v1`
- **OpenAI API Key**: `fake`

### Chat Output
- Displays the final LLM response in the Playground
- **Note**: Qwen3 outputs `<think>...</think>` reasoning tags before the actual answer. This is a built-in feature of Qwen3 and cannot be disabled via LlamaStack's API.

## Code Modifications Required

### A. OpenAI Embeddings (Both ingestion + search)

Both OpenAI Embeddings components require **two code fixes**. Click the `</>` (Code) button on each one.

### Fix 1: Custom Model Name (~line 40)

The model field is a locked `DropdownInput` with only OpenAI model names. Change it to free-text:

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

### Fix 2: Disable Context Length Check (~line 75, inside `build_embeddings` method)

**Why this is needed**: When TikToken is disabled (via the UI toggle), LangChain falls back to loading a HuggingFace tokenizer to check embedding context length:

```
tiktoken_enabled=True  → uses OpenAI tiktoken (sends token IDs → LlamaStack rejects with "Input should be a valid string")
tiktoken_enabled=False → falls back to HuggingFace AutoTokenizer.from_pretrained("granite-embedding-125m") → fails because it's not a valid HF model ID
```

The fix: skip the client-side context length check entirely. The Granite model on LlamaStack handles tokenization server-side.

```python
# Find this line:
            tiktoken_enabled=self.tiktoken_enable,

# Add one line right after it:
            tiktoken_enabled=self.tiktoken_enable,
            check_embedding_ctx_length=False,
```

> **Important**: Both fixes must be applied to EACH OpenAI Embeddings component independently. Editing one does not affect the other.

### B. OpenAI Model (LLM)

The OpenAI Model component also has a locked dropdown for model names. Click `</>` and make two edits:

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

Langflow leaks the `display_name` ("Model Name") into `model_kwargs`, causing:
`AsyncCompletions.create() got an unexpected keyword argument 'Model Name'`

```python
# Find this line:
        model_kwargs = self.model_kwargs or {}

# Add right after it:
        model_kwargs = self.model_kwargs or {}
        model_kwargs.pop("Model Name", None)
```

> **Why this happens**: Langflow's internal mechanism sometimes passes the field's `display_name` (with a space: "Model Name") as a keyword argument in `model_kwargs`. The OpenAI client doesn't recognize this and throws a `TypeError`.

## Connection Map

### Ingestion Flow

| From Component | From Port | To Component | To Port |
|---|---|---|---|
| Docling Serve | Files | Export DoclingDocument | Data or DataFrame |
| Export DoclingDocument | Exported data | Split Text | Input |
| Split Text | Chunks | Milvus (ingest) | Ingest Data |
| OpenAI Embeddings (ingest) | Embeddings | Milvus (ingest) | Embedding |

### Retriever Flow

| From Component | From Port | To Component | To Port |
|---|---|---|---|
| Chat Input | Message | Milvus (search) | Search Query |
| OpenAI Embeddings (search) | Embeddings | Milvus (search) | Embedding |
| Milvus (search) | Search Results | Parser | Input |
| Parser | Output | Prompt | `{context}` template variable |
| Chat Input | Message | Prompt | `{question}` template variable |
| Prompt | Prompt Message | OpenAI Model | Input |
| OpenAI Model | Text Response | Chat Output | Text |

## All Code Edits Summary

| Component | Fix | Change |
|-----------|-----|--------|
| **OpenAI Embeddings** (x2) | Fix 1 | `DropdownInput` → `MessageTextInput` for model |
| **OpenAI Embeddings** (x2) | Fix 2 | Add `check_embedding_ctx_length=False` |
| **OpenAI Model** (x1) | Fix 1 | `DropdownInput` → `StrInput` for model_name |
| **OpenAI Model** (x1) | Fix 2 | Add `model_kwargs.pop("Model Name", None)` |

## Common Pitfalls

1. **Wrong embedding component**: Use **OpenAI Embeddings** (Bundles > OpenAI), not core "Embedding Model" or "HuggingFace Embeddings Inference"
2. **Embedding model name locked**: Must edit component code to change `DropdownInput` → `MessageTextInput` (Embedding Fix 1)
3. **LLM model name locked**: Must edit component code to change `DropdownInput` → `StrInput` (LLM Fix 1)
4. **HuggingFace tokenizer error**: Must add `check_embedding_ctx_length=False` in code (Embedding Fix 2)
5. **TikToken sends integers**: Must toggle TikToken Enable to `false` in advanced settings
6. **Leaked 'Model Name' kwarg**: Must add `model_kwargs.pop("Model Name", None)` (LLM Fix 2)
7. **Parser mode**: Must use **Stringify**, not Parser (Milvus returns a list of Data objects)
8. **Milvus collection name**: No hyphens (`cams-docs` fails, use `cams_docs_v2`)
9. **Milvus URI typo**: Must start with `http://` (not `ttp://`)
10. **Embedding port**: OpenAI Embeddings must connect to Milvus's **Embedding** port, not **Ingest Data**
11. **Missing Export DoclingDocument**: Without it, Docling's `doc` field causes `unrecognized dtype` error
12. **Both Embedding components need editing**: Code fixes are per-component, not global
13. **Qwen3 `<think>` tags**: Cannot be disabled via LlamaStack — the answer appears after `</think>`
