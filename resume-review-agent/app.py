import io
import streamlit as st
import pypdf
from crewai import Agent, Crew, Process, Task, LLM

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Resume Review Agent",
    page_icon="📄",
    layout="wide"
)

# --- Helper: PDF Text Extraction ---
def extract_text_from_pdf(uploaded_file) -> tuple[str | None, str | None]:
    """
    Extracts selectable text from a PDF file using pypdf.
    Returns (extracted_text, error_message).
    """
    try:
        pdf_reader = pypdf.PdfReader(uploaded_file)
        if len(pdf_reader.pages) == 0:
            return None, "The uploaded PDF document contains no pages."

        extracted_text = []
        for index, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_text.append(page_text.strip())

        full_text = "\n\n".join(extracted_text).strip()
        if not full_text:
            return None, (
                "No selectable text could be extracted from this PDF. "
                "If this is a scanned document or an image, please convert it "
                "using OCR or paste the text directly into the text tab."
            )
        return full_text, None
    except Exception as exc:
        return None, f"An error occurred while reading the PDF: {str(exc)}"

# --- Helper: CrewAI Setup & Execution ---
def run_resume_review(resume_content: str, job_description: str, groq_api_key: str, groq_model: str):
    """
    Sets up a single-agent CrewAI pipeline and returns the structured evaluation.
    """
    # Format model string for CrewAI / LiteLLM integration
    model_identifier = f"groq/{groq_model}" if not groq_model.startswith("groq/") else groq_model

    llm = LLM(
        model=model_identifier,
        api_key=groq_api_key,
        temperature=0.1
    )

    evaluator_agent = Agent(
        role="Principal Technical Recruiter and Career Auditor",
        goal=(
            "Thoroughly analyze a candidate's resume strictly against a targeted job description. "
            "Never assume, extrapolate, or fabricate unmentioned qualifications."
        ),
        backstory=(
            "You are a meticulous, objective recruitment analyst with over 15 years of technical "
            "hiring experience. Your evaluations are prized for factual precision. You never assume "
            "a candidate has a skill unless explicitly verified in their supplied text. If an "
            "expected qualification is missing or ambiguous, you mark it clearly as missing or unverified."
        ),
        llm=llm,
        verbose=False,
        memory=False,
        allow_delegation=False
    )

    evaluation_task = Task(
        description=(
            "Conduct an in-depth, strictly grounded evaluation of the provided RESUME against the target JOB DESCRIPTION.\n\n"
            "STRICT GROUNDING RULES:\n"
            "1. Base your evaluation ONLY on the exact text provided in the resume.\n"
            "2. NEVER invent, infer, or hallucinate projects, certifications, metrics, or years of experience.\n"
            "3. If a skill, degree, or experience is not mentioned in the resume, you MUST classify it as "
            "'Unknown / Not Demonstrated' or 'Missing'.\n\n"
            "INPUT DATA:\n"
            "--- TARGET JOB DESCRIPTION ---\n{job_description}\n\n"
            "--- CANDIDATE RESUME ---\n{resume_content}\n\n"
            "REQUIRED OUTPUT STRUCTURE:\n"
            "Please deliver your assessment using the following Markdown headings:\n\n"
            "## 1. Match Summary\n"
            "- High-level alignment overview and overall fit rating (Low / Moderate / Strong).\n\n"
            "## 2. Skills Found\n"
            "- Explicitly stated technical and domain skills that match the target role.\n\n"
            "## 3. Missing Requirements\n"
            "- Required or preferred qualifications from the job description that do not appear in the resume.\n\n"
            "## 4. Unclear / Not Demonstrated\n"
            "- Qualifications vaguely hinted at but lacking concrete metrics, tools, or proof.\n\n"
            "## 5. Experience Gaps\n"
            "- Deficits in seniority, tenure, project scale, or domain ownership.\n\n"
            "## 6. Education / Qualification Gaps\n"
            "- Degree, accreditation, or certification discrepancies.\n\n"
            "## 7. Resume Improvements\n"
            "- Concrete, bulleted recommendations to reframe existing experience using the XYZ formula (Accomplished [X] as measured by [Y], by doing [Z]).\n\n"
            "## 8. Keywords to Consider\n"
            "- ATS-relevant terms from the job description that the candidate genuinely possesses but omitted.\n\n"
            "## 9. Priority Action Plan\n"
            "- A 3-step prioritized checklist for the candidate to improve their application immediately."
        ),
        expected_output="A structured markdown report covering all nine evaluation sections with complete factual adherence.",
        agent=evaluator_agent
    )

    crew = Crew(
        agents=[evaluator_agent],
        tasks=[evaluation_task],
        process=Process.sequential,
        verbose=False
    )

    return crew.kickoff(inputs={
        "job_description": job_description,
        "resume_content": resume_content
    })

# --- UI Header & Privacy Notice ---
st.title("📄 AI Resume Review Agent")
st.caption("Powered by CrewAI, Groq, and Streamlit")

st.info(
    "🔒 **Privacy Notice**: Uploaded resumes and job descriptions are processed in memory and "
    "sent securely to the configured Groq LLM endpoint solely for evaluation. Data is never permanently "
    "retained or stored on this server.",
    icon="ℹ️"
)

# --- Check Configuration Secrets ---
groq_api_key = st.secrets.get("GROQ_API_KEY")
groq_model = st.secrets.get("GROQ_MODEL", "llama-3.3-70b-versatile")

if not groq_api_key:
    st.error(
        "⚠️ **Missing Configuration**: `GROQ_API_KEY` was not found in your Streamlit secrets. "
        "Please configure it in `.streamlit/secrets.toml` locally or in your Streamlit Cloud settings."
    )
    st.stop()

# --- Input Section ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Candidate Resume")
    resume_tab_upload, resume_tab_paste = st.tabs(["Upload PDF", "Paste Plain Text"])
    
    extracted_resume_text = ""
    
    with resume_tab_upload:
        uploaded_pdf = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])
        if uploaded_pdf is not None:
            text_result, err = extract_text_from_pdf(uploaded_pdf)
            if err:
                st.error(err)
            else:
                extracted_resume_text = text_result
                st.success(f"Extracted {len(extracted_resume_text.split())} words from '{uploaded_pdf.name}'.")
    
    with resume_tab_paste:
        pasted_resume_text = st.text_area(
            "Paste full resume content",
            height=300,
            placeholder="Paste raw resume text here..."
        )
        if pasted_resume_text.strip():
            extracted_resume_text = pasted_resume_text.strip()

with col2:
    st.subheader("2. Target Job Description")
    job_description_input = st.text_area(
        "Paste the job description you are targeting",
        height=375,
        placeholder="Paste requirements, responsibilities, and qualifications..."
    )

# --- Submission & Validation ---
st.divider()

if st.button("Evaluate Application Match", type="primary", use_container_width=True):
    # Validation checks
    if not extracted_resume_text.strip():
        st.warning("Please upload a readable PDF or paste your resume text before running the review.")
    elif not job_description_input.strip():
        st.warning("Please provide a target job description to evaluate against.")
    elif len(extracted_resume_text.strip()) < 50:
        st.warning("The provided resume text is too short. Please provide a complete resume.")
    elif len(job_description_input.strip()) < 50:
        st.warning("The job description is too brief. Please paste the full job posting.")
    else:
        with st.spinner("Analyzing alignment, identifying qualification gaps, and formulating feedback..."):
            try:
                review_result = run_resume_review(
                    resume_content=extracted_resume_text,
                    job_description=job_description_input,
                    groq_api_key=groq_api_key,
                    groq_model=groq_model
                )
                
                st.subheader("📋 Evaluation Results")
                st.markdown(str(review_result))
                
            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "429" in error_msg:
                    st.error("Groq API rate limit exceeded. Please wait a moment before trying again.")
                elif "unauthorized" in error_msg or "invalid api key" in error_msg or "401" in error_msg:
                    st.error("Authentication failed. Please verify that your `GROQ_API_KEY` is active and correct.")
                elif "timeout" in error_msg:
                    st.error("The request to the model timed out. Please try again with slightly shorter input text.")
                else:
                    st.error(f"An unexpected error occurred during analysis: {str(e)}")
