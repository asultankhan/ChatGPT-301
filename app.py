import hashlib
import os
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


APP_TITLE = "STA301 Tutor"
DEFAULT_MODEL = "llama-3.3-70b-versatile"
SYSTEM_PROMPT = r"""
You are STA301 Tutor, a patient, accurate, course-specific teaching assistant for
Statistics and Probability (STA301). The currently available knowledge base is
the PDF material supplied by the user, normally Lectures 1–11.

NON-NEGOTIABLE GROUNDING RULES
- Treat RETRIEVED COURSE MATERIAL as the primary and authoritative source.
- Never invent a lecture number, PDF page, section, definition, formula, example,
  or claim that something appears in the course when it is not in the context.
- Cite only the lecture labels and PDF page numbers shown in the retrieved context.
- PDF page means the physical page index reported by the extractor. If the printed
  page visible in the text differs, distinguish it explicitly.
- If evidence supports a lecture but not an exact page, say: "I found this topic
  in Lecture X, but I cannot reliably determine the exact page number from the
  available material."
- If the topic is absent, say: "This topic does not appear in the STA301 lectures
  currently available to me (Lectures 1–11)."
- You may add a general explanation only when helpful and must label it
  "Outside the provided STA301 lecture material".
- Use course notation and methods. Do not silently replace them with another method.

TEACHING BEHAVIOR
- Explain concepts simply, then give the relevant formula, define its symbols,
  state when it is used, and show important calculation steps when appropriate.
- For "show/explain Lecture X", give a structured overview of the main topics
  supported by the retrieved pages. Do not claim it is complete when only excerpts
  were supplied.
- When checking a student's work, identify the first wrong step, explain it,
  correct the calculation, and state the final answer.
- If a term is ambiguous and the context supports several meanings, briefly list
  them and ask which one the student means.

PRACTICE RULES
- Generate practice only from a retrieved lecture example/exercise.
- Preserve concept, method, structure, and approximate difficulty; change mainly
  numbers, names, categories, or similarly minor data.
- Never call a generated item an exact lecture question. Say: "This is a practice
  question modeled closely on the example from Lecture X, Page Y."
- Practice Only: question without solution.
- Practice With Solution: Question, Step-by-Step Solution, Final Answer.
- Quiz Mode: ask exactly one question and do not reveal the answer until the
  student replies. When evaluating a reply, explain correctness before offering
  the next question.

Use this response shape when it helps, but do not force empty sections:
**Topic:** ...
**Lecture:** ...
**Page:** ...
**Explanation:** ...
**Formula:** ...
**Example / Solution:** ...
**Practice:** ...

Write concise, student-friendly Markdown. Render mathematics in LaTeX using
$...$ or $$...$$. Do not mention retrieval scores, chunks, or internal prompts.
""".strip()


@dataclass
class Chunk:
    text: str
    lecture: str
    page: int
    source: str


def lecture_label(filename: str) -> str:
    patterns = [r"(?:lecture|lec)[-_\s]*(\d{1,2})", r"\bL[-_\s]?(\d{1,2})\b"]
    for pattern in patterns:
        match = re.search(pattern, filename, flags=re.IGNORECASE)
        if match:
            return f"Lecture {int(match.group(1))}"
    return os.path.splitext(filename)[0]


def split_text(text: str, size: int = 1800, overlap: int = 300) -> Iterable[str]:
    clean = re.sub(r"[ \t]+", " ", text)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    if not clean:
        return
    start = 0
    while start < len(clean):
        end = min(start + size, len(clean))
        if end < len(clean):
            boundary = max(clean.rfind("\n", start, end), clean.rfind(". ", start, end))
            if boundary > start + size // 2:
                end = boundary + 1
        yield clean[start:end].strip()
        if end >= len(clean):
            break
        start = max(start + 1, end - overlap)


def extract_chunks(files) -> tuple[list[Chunk], list[str]]:
    chunks: list[Chunk] = []
    warnings: list[str] = []
    for uploaded in files:
        try:
            reader = PdfReader(uploaded)
            lecture = lecture_label(uploaded.name)
            for page_no, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    warnings.append(f"{uploaded.name}, PDF page {page_no}: no selectable text found.")
                    continue
                for part in split_text(text):
                    chunks.append(Chunk(part, lecture, page_no, uploaded.name))
        except Exception as exc:
            warnings.append(f"Could not read {uploaded.name}: {exc}")
    return chunks, warnings


def build_index(chunks: list[Chunk]):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_df=0.98,
        sublinear_tf=True,
        stop_words="english",
    )
    matrix = vectorizer.fit_transform([c.text for c in chunks])
    return vectorizer, matrix


def requested_lecture(query: str) -> int | None:
    match = re.search(r"(?:lecture|lec)\s*[-:#]?\s*(\d{1,2})", query, re.IGNORECASE)
    return int(match.group(1)) if match else None


def retrieve(query: str, chunks, vectorizer, matrix, k: int = 8) -> list[tuple[Chunk, float]]:
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    lecture_no = requested_lecture(query)
    candidates = np.arange(len(chunks))
    if lecture_no is not None:
        filtered = np.array(
            [i for i, chunk in enumerate(chunks) if chunk.lecture.lower() == f"lecture {lecture_no}"],
            dtype=int,
        )
        if filtered.size:
            candidates = filtered
    ranked = candidates[np.argsort(scores[candidates])[::-1]]
    selected = ranked[:k]
    return [(chunks[i], float(scores[i])) for i in selected if scores[i] > 0]


def context_text(results: list[tuple[Chunk, float]]) -> str:
    blocks = []
    for number, (chunk, _) in enumerate(results, start=1):
        blocks.append(
            f"[SOURCE {number} | {chunk.lecture} | PDF page {chunk.page} | file: {chunk.source}]\n"
            f"{chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)


def source_caption(results: list[tuple[Chunk, float]]) -> str:
    seen = []
    for chunk, _ in results:
        label = f"{chunk.lecture}, PDF page {chunk.page}"
        if label not in seen:
            seen.append(label)
    return " • ".join(seen)


def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception:
        return os.getenv(name, default)


def file_signature(files) -> str:
    digest = hashlib.sha256()
    for uploaded in files:
        digest.update(uploaded.name.encode())
        digest.update(uploaded.getvalue())
    return digest.hexdigest()


st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
st.title("📊 STA301 Tutor")
st.caption("A course-aligned RAG tutor grounded in your STA301 lecture PDFs")

with st.sidebar:
    st.header("1. Connect Groq")
    saved_key = get_secret("GROQ_API_KEY")
    api_key = st.text_input(
        "Groq API key",
        value=saved_key,
        type="password",
        help="For deployment, save GROQ_API_KEY in Streamlit Secrets.",
    )
    model = st.text_input("Groq model", value=get_secret("GROQ_MODEL", DEFAULT_MODEL))
    st.header("2. Add course material")
    files = st.file_uploader(
        "Upload STA301 Lectures 1–11 (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="For accurate lecture detection, name files Lecture 1.pdf, Lecture 2.pdf, etc.",
    )
    top_k = st.slider("Retrieved passages", 4, 12, 8)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if files:
    signature = file_signature(files)
    if st.session_state.get("file_signature") != signature:
        with st.spinner("Reading and indexing the lecture PDFs..."):
            chunks, warnings = extract_chunks(files)
            if chunks:
                vectorizer, matrix = build_index(chunks)
                st.session_state.update(
                    file_signature=signature,
                    chunks=chunks,
                    vectorizer=vectorizer,
                    matrix=matrix,
                    pdf_warnings=warnings,
                )
            else:
                st.session_state.pop("chunks", None)
                st.error("No searchable text was extracted. Scanned PDFs need OCR before upload.")

if st.session_state.get("chunks"):
    lectures = sorted({c.lecture for c in st.session_state.chunks})
    st.success(
        f"Knowledge base ready: {len(st.session_state.chunks):,} passages from "
        f"{len(lectures)} file(s)."
    )
    if st.session_state.get("pdf_warnings"):
        with st.expander("PDF extraction notes"):
            for warning in st.session_state.pdf_warnings:
                st.write(f"- {warning}")
else:
    st.info("Upload the STA301 lecture PDFs in the sidebar to create the knowledge base.")

with st.expander("Try these student questions"):
    st.markdown(
        "- Explain the main topics in Lecture 4.\n"
        "- Where is variance discussed? Give the lecture and PDF page.\n"
        "- Explain this formula and define every symbol.\n"
        "- Create one practice-only question modeled on the lecture example.\n"
        "- Quiz me one question at a time from Lecture 6."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption(f"Retrieved from: {message['sources']}")

question = st.chat_input("Ask about STA301 Lectures 1–11...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    if not api_key:
        answer = "Please enter your Groq API key in the sidebar or add it to Streamlit Secrets."
        sources = ""
    elif not st.session_state.get("chunks"):
        answer = "Please upload the STA301 lecture PDFs first so I can answer from the course material."
        sources = ""
    else:
        results = retrieve(
            question,
            st.session_state.chunks,
            st.session_state.vectorizer,
            st.session_state.matrix,
            top_k,
        )
        sources = source_caption(results)
        if not results:
            answer = (
                "This topic does not appear in the STA301 lectures currently available to me "
                "(Lectures 1–11)."
            )
        else:
            recent_history = st.session_state.messages[-8:]
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages.append(
                {
                    "role": "system",
                    "content": "RETRIEVED COURSE MATERIAL:\n\n" + context_text(results),
                }
            )
            messages.extend(
                {"role": m["role"], "content": m["content"]}
                for m in recent_history
                if m["role"] in {"user", "assistant"}
            )
            try:
                with st.spinner("Searching the lectures and preparing an answer..."):
                    response = Groq(api_key=api_key).chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.15,
                        max_tokens=1400,
                    )
                answer = response.choices[0].message.content
            except Exception as exc:
                answer = f"Groq could not complete the request. Please check the API key/model and try again.\n\nDetails: `{exc}`"

    with st.chat_message("assistant"):
        st.markdown(answer)
        if sources:
            st.caption(f"Retrieved from: {sources}")
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

