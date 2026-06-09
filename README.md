# Gen-AI Course Assistant — Multi-Agent Study System

A lecture-grounded **study assistant** for a university course on Generative AI / LLMs. It
generates difficulty-controlled **MCQ exams** grounded in the course slides, grades a student's
answer, explains *why* the answer is right/wrong with **slide citations**, and recommends extra
learning resources. It ties together three components built in separate notebooks — a fine-tuned
generator, a RAG retriever, and a set of tools — into one **LangGraph** multi-agent pipeline.

> **Disclaimer:** All questions and explanations are AI-generated from course lecture notes and
> may contain errors. Always verify against the original slides and your instructor.

---

## 1. Repository layout

```
GenAI Project/
├── MCQ_MultiAgent_LangGraph.ipynb   # ★ THE INTEGRATED SYSTEM (run this)
├── RAG/
│   ├── Hybrid_RAG+LLM.ipynb         # builds the RAG corpus + retriever
│   ├── embeddings.npy               # 664 × 1024 slide vectors (L2-normalised)  ← used by §5
│   ├── chunks_metadata.json         # 664 slide chunks (id, text, metadata)     ← used by §5
│   └── qdrant_db_backup.zip         # same vectors in a Qdrant store (production option)
├── Fine-Tuning/
│   └── MCQ_SFT_Qwen3_4B.ipynb       # QLoRA fine-tune of Qwen3-4B on the MCQ bank
└── Dataset/
    ├── Lectures/                    # 9 lecture PDFs (source material)
    ├── Lectures Note/               # 9 slide-explanation JSONs  → RAG corpus + citations
    └── MCQ Exams/                   # llm_mcq_exam_bank.xlsx (2,111 MCQs) → fine-tuning + tool
```

The fine-tuned adapter is hosted on Hugging Face: `Marryam03/mcq-sft-qwen3-4b-lora-v2`.

---

## 2. How the pieces connect (architecture)

```
              ┌──────────── LangGraph StateGraph (exam generation) ────────────┐
  topic  ──▶  │ Planner ─▶ Retriever ─▶ Generator ─▶ Critic ─┐                  │
              │ (Gemini)   (hybrid RAG)  (fine-tuned   (rules)│ hard fail?       │
              │                           Qwen3-4B)          └─▶ loop (≤3) ──▶  Finalize │ ──▶ MCQ
              └──────────────────────────────────────────────────────────────┘   (shuffle + RAG citations)

  student answer ──▶ grade ──▶ Feedback Tutor (RAG-grounded, cited)                                  └─ on wrong answer ─▶ Resource Recommender ─▶ youtube_search (tool call)
```

**Agents / roles**

| Agent | Engine | Job |
|---|---|---|
| **Curriculum Planner** | Gemini | Turns a topic into a JSON spec (concept, difficulty, Bloom level, question type, retrieval query). |
| **Retriever** | Hybrid RAG | Dense (Qwen3-Embedding-0.6B) + BM25 fused with **RRF** over the lecture notes. |
| **MCQ Generator** | Fine-tuned Qwen3-4B | Produces the MCQ JSON *structure* from the retrieved slide context. |
| **Critic / Examiner** | Deterministic rules | Rejects bad questions (wrong option count, duplicates, length bias) and auto-fixes minor issues; drives the regenerate loop. |
| **Feedback Tutor** | Gemini + RAG | Re-authors the why-right/why-wrong explanation **from retrieved slide text**, with `[L#-S#]` citations. |
| **Resource Recommender** | Gemini + tool | Calls `youtube_search` itself via function calling to suggest videos for the missed concept. |

---

## 3. Required LLM topics (rubric coverage)

| Topic | Where | What it does |
|---|---|---|
| **Prompt design** | Per-agent system prompts (role/style/constraints) + structured JSON specs for the generator. |
| **RAG** | Hybrid dense+BM25+RRF retriever over lecture notes; grounds generation, overrides citations, and authors feedback. |
| **Fine-tuning / PEFT** | QLoRA on Qwen3-4B over the MCQ bank; produces valid, grounded MCQ JSON. |
| **Tools / function calling** | `question_bank_lookup`, `slide_lookup_tool`, `youtube_search`; Gemini calls `youtube_search` automatically. |
| **Multi-agent** | LangGraph graph with a Generator↔Critic revise loop, plus the Tutor+Recommender collaboration. |
| **Evaluation** | Scorecard + single-agent vs multi-agent; base-vs-fine-tuned in the SFT notebook. |
| **Ethics & safety** | Disclaimer on every output, refusal guard via retrieval relevance, risk discussion. |

---

## 4. Models, APIs, frameworks

- **Generator:** `Marryam03/mcq-sft-qwen3-4b-lora-v2` (Qwen3-4B-Base + LoRA, trained with Unsloth/TRL).
- **Embeddings (RAG):** `Qwen/Qwen3-Embedding-0.6B` (dense), `rank_bm25` (lexical), RRF fusion.
- **Reasoning agents:** Google **Gemini** `gemini-3.1-flash-lite` via the `google-genai` SDK
  (automatic function calling).
- **Orchestration:** **LangGraph** `StateGraph`.
- **Vector store:** Qdrant (production); the notebook reads the exported `embeddings.npy` directly
  for portability (identical cosine results).

---

## 5. Demo & evaluation

- **Grounded RAG example:** a generated question with `evidence_slide_ids` pulled from the retriever.
- **Multi-agent example:** the Critic loop log (`attempts`, `critic.reasons`).
- **Tool-use example:** the Resource Recommender's `youtube_search` call on a wrong answer.
- **Baselines:** single-agent (raw generator) vs multi-agent (full graph) on valid-rate,
  duplicate-rate, length-bias, and grounded-rate; base-vs-fine-tuned lives in the SFT notebook.

---

## 7. Ethics, safety, limitations

- **Disclaimer** attached to every student-facing output.
- **Refusal:** `on_topic()` refuses topics with no sufficiently similar lecture slide;
  prompts decline non-course questions and won't give answers without the explanation.
- **Hallucination mitigation:** questions grounded in retrieved slides; citations overridden by the
  retriever; feedback re-authored from slide text, never the generator's draft.
- **Bias/position:** Critic rejects length bias; `shuffle_mcq_output` enforces uniform answer position.
- **Privacy:** no student PII stored; corpus is course material only.
- **Limitations:** 4B generator; quality depends on retrieval; YouTube tool needs an API key for
  live results; difficulty labels are approximate.
