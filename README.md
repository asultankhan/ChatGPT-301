# STA301 Tutor — Groq + Streamlit RAG Chatbot

STA301 Tutor is a page-aware Retrieval-Augmented Generation (RAG) chatbot for
Statistics and Probability. Students upload Lectures 1–11 as PDFs, and the app
retrieves relevant passages before asking a Groq model to answer. Responses cite
the lecture and physical PDF page supplied to the model.

## Features

- Answers from uploaded STA301 PDFs rather than an unrestricted model response.
- Reports lecture number and physical PDF page when available.
- Explains formulas, notation, and examples in student-friendly steps.
- Produces practice questions closely modeled on retrieved lecture examples.
- Supports practice-only, worked-solution, and one-question-at-a-time quiz modes.
- Checks student solutions and explains the first incorrect step.
- Refuses to invent unsupported lecture/page references.

## Project files

```text
sta301-rag-chatbot/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── secrets.toml.example
```

The lecture PDFs are intentionally not committed. Upload them through the app.
Name them `Lecture 1.pdf`, `Lecture 2.pdf`, …, `Lecture 11.pdf` so the app can
identify lecture numbers reliably.

## Run locally

1. Install Python 3.10 or newer.
2. In this project folder, create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add
   your Groq API key. Never commit `secrets.toml`.
5. Run:

   ```bash
   streamlit run app.py
   ```

You may get a Groq API key from the Groq Console. The default model can be
changed from the sidebar or by changing `GROQ_MODEL` in Streamlit Secrets.

## Upload to GitHub

1. Create a new GitHub repository, for example `sta301-rag-chatbot`.
2. Upload all project files and folders except `.streamlit/secrets.toml`.
3. Suggested commit message: `Build STA301 page-aware RAG tutor with Groq`

## Deploy on Streamlit Community Cloud

1. Open Streamlit Community Cloud and select **Create app**.
2. Choose the GitHub repository and branch.
3. Set the main file path to `app.py`.
4. Open **Advanced settings → Secrets** and add:

   ```toml
   GROQ_API_KEY = "your_real_key"
   GROQ_MODEL = "llama-3.3-70b-versatile"
   ```

5. Deploy the app. Upload Lectures 1–11 through the sidebar after it starts.

## Accuracy notes

- The displayed page is the physical PDF page index, beginning with 1. It may
  differ from a page number printed inside the lecture document.
- Image-only scanned PDFs must be OCR-processed before upload.
- The app uses TF-IDF retrieval. It is lightweight and inexpensive for
  Streamlit deployment, though semantic embeddings can be added later.
- Uploaded PDFs and the in-memory index last only for the current Streamlit
  session. This avoids committing copyrighted course material or exposing it in
  a public repository.

## Security

Never write a real API key in `app.py`, GitHub, screenshots, or a public file.
Store it only in Streamlit Secrets or a local untracked `secrets.toml` file.

