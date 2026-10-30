# AI Resume Review Agent

A production-ready, single-agent AI application built with **Streamlit**, **CrewAI**, and **Groq**. The agent evaluates a candidate's resume against a target job description, pinpointing verifiable skills, experience gaps, and high-impact resume improvements without hallucinating unmentioned qualifications.

## Features

- **Dual Resume Input**: Accepts both PDF uploads (parsed via `pypdf`) and plain-text pasting.
- **Hallucination Prevention**: Strict agent guardrails prohibit assumptions or ungrounded claims. Missing items are marked as "Unknown / Not Demonstrated".
- **Structured Output**: Delivers an actionable 9-part breakdown including an ATS keyword check and a priority action plan.
- **Ultra-Fast LLM Inference**: Integrated with Groq's high-speed API.

## Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/resume-review-agent.git
   cd resume-review-agent
   ```

2. **Create and activate a Python 3.11 virtual environment**:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Secrets**:
   Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`:
   ```bash
   mkdir -p .streamlit
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```
   Open `.streamlit/secrets.toml` and paste your actual Groq API key:
   ```toml
   GROQ_API_KEY = "gsk_..."
   GROQ_MODEL = "llama-3.3-70b-versatile"
   ```

5. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```
