import streamlit as st
import re
import random
import textstat
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
from io import BytesIO
from PIL import Image
import time

from PyPDF2 import PdfReader
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction import text

# Set page config first
st.set_page_config(page_title="ResuGenius", layout="wide")

# === Load logo ===
st.image("file-FtDzaG9mTPrpYZvHAFrVaj", width=120)
st.markdown("<h1 style='color:#102B5A; font-size: 40px;'>ResuGenius: AI Resume Screening System</h1>", unsafe_allow_html=True)
st.markdown("*Your skill. Your story. Verified.*", unsafe_allow_html=True)
st.divider()

# === Sentiment model ===
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

sentiment_pipeline = load_sentiment_pipeline()

# === Utility Functions ===
stop_words = text.ENGLISH_STOP_WORDS

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    return "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])

def clean_text(txt):
    return " ".join([w for w in txt.lower().split() if w not in stop_words])

def get_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return result['label'].lower(), float(result['score'])

def get_readability_score(text):
    return textstat.flesch_reading_ease(text)

def suggest_keywords(jd, resume):
    jd_words = set(jd.lower().split())
    resume_words = set(resume.lower().split())
    return list(jd_words - resume_words)[:5]

def get_quiz_questions():
    return {
        "Tech": [
            {"question": "What does HTML stand for?", "options": ["HyperText Markup Language", "Hyper Tool Multi Language", "Home Tool Markup Language"], "answer": "HyperText Markup Language"},
            {"question": "What does CSS control?", "options": ["Layout and Style", "Database", "Server"], "answer": "Layout and Style"},
            # Add 8 more
        ],
        "HR": [
            {"question": "Which task is an HR responsibility?", "options": ["Hiring", "Coding", "Wiring"], "answer": "Hiring"},
            # Add 9 more
        ],
        # Add more domains
    }

# === Consent Section ===
st.markdown("### ☑️ Consent")
consent = st.checkbox("I understand that my data is not stored and I am responsible for submitting my report.")
if not consent:
    st.stop()

# === Resume Upload Section ===
st.divider()
st.markdown("### 📄 Resume Upload + Job Description")
name = st.text_input("Your full name")
domain_selector = st.selectbox("Select the domain you're applying for", list(get_quiz_questions().keys()))
job_description = st.text_area("Paste the job description here")
uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

if uploaded_file and job_description:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.session_state.resume_text = resume_text
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vectors = tfidf.fit_transform([clean_text(job_description), clean_text(resume_text)])
    match_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    
    # Encouragement mode (soften low scores)
    adjusted_score = max(match_score, 0.45)
    st.success(f"Resume Match Score: {adjusted_score:.2f}")

    readability = get_readability_score(resume_text)
    st.info(f"Readability Score: {readability:.1f}")
    keywords = suggest_keywords(job_description, resume_text)
    st.write(f"Suggested Keywords: {', '.join(keywords)}")

# === Skill Quiz Section ===
st.divider()
st.markdown("### ⏱️ Skill Quiz")
questions = random.sample(get_quiz_questions()[domain_selector], min(10, len(get_quiz_questions()[domain_selector])))
quiz_score = 0

for i, q in enumerate(questions):
    st.subheader(f"Question {i+1}: {q['question']}")
    with st.container():
        options = q["options"]
        user_answer = st.radio("Choose your answer:", options, key=f"q{i}")
        placeholder = st.empty()
        for sec in range(5, 0, -1):
            placeholder.markdown(f"*Time left: {sec} sec*")
            time.sleep(1)
        placeholder.empty()
        if user_answer == q["answer"]:
            st.success("Correct!")
            quiz_score += 1
        elif user_answer:
            st.error("Incorrect!")
        else:
            st.warning("No answer selected.")

st.info(f"Final Quiz Score: {quiz_score}/10")

# === Skill Certificate ===
if quiz_score >= 6:
    st.success("✅ Skill Verified!")
    if st.button("Download Your Certificate"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "CERTIFICATE OF SKILL VERIFICATION", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.ln(10)
        pdf.multi_cell(0, 10, f"This is to certify that {name} has successfully demonstrated proficiency in the {domain_selector} domain with a score of {quiz_score}/10.")
        pdf.multi_cell(0, 10, f"This achievement reflects that {name} is capable enough to work and contribute effectively in this field.")
        pdf.ln(10)
        pdf.cell(0, 10, f"{datetime.today().strftime('%B %d, %Y')}", ln=True)
        buffer = BytesIO()
        pdf.output(buffer)
        st.download_button("Download Certificate (PDF)", data=buffer.getvalue(), file_name=f"{name}_certificate.pdf", mime="application/pdf")
else:
    st.warning("You didn't qualify for a certificate yet — but you’re on the right track!")

    # === Emotional Insights Section ===
st.divider()
st.markdown("### 💬 Emotional Intelligence Insights")

pitch = st.text_area("Why should we hire you?")
motivation = st.text_area("What drives you professionally?")
achievement = st.text_area("What achievement are you most proud of?")
failure = st.text_area("How do you handle failure?")

if all([pitch, motivation, achievement, failure]):
    sent_data = {
        "Pitch": get_sentiment(pitch),
        "Motivation": get_sentiment(motivation),
        "Achievement": get_sentiment(achievement),
        "Failure": get_sentiment(failure)
    }

    # EQ Profile
    def classify_trait(score):
        if score > 0.8: return "High"
        elif score > 0.6: return "Medium"
        else: return "Low"

    eq_profile = {
        "Resilience": classify_trait(sent_data["Failure"][1]),
        "Self-awareness": classify_trait(sent_data["Motivation"][1]),
        "Confidence in Expression": classify_trait((sent_data["Pitch"][1] + sent_data["Achievement"][1])/2)
    }

    st.markdown("#### EQ Profile")
    for trait, level in eq_profile.items():
        st.markdown(f"- *{trait}*: {level}")

    st.markdown("#### Sentiment Breakdown")
    for label, (sentiment, score) in sent_data.items():
        st.markdown(f"- *{label}*: {sentiment.title()} ({score:.0%})")

    # Bar Chart
    fig1, ax1 = plt.subplots()
    ax1.bar(sent_data.keys(), [s[1] for s in sent_data.values()])
    ax1.set_title("Confidence in Emotional Tone")
    ax1.set_ylim(0, 1)
    st.pyplot(fig1)

    # Quiz Chart
    fig2, ax2 = plt.subplots()
    ax2.pie([quiz_score, 10 - quiz_score], labels=["Correct", "Incorrect"], colors=["#4CAF50", "#FF6F61"], autopct='%1.1f%%')
    ax2.set_title("Quiz Result Breakdown")
    st.pyplot(fig2)

    # Recommendations
    st.divider()
    st.markdown("### 📚 Recommendations & Resources")
    if quiz_score < 6:
        st.markdown("You can improve further! Here are some helpful resources:")
        course_list = {
            "Tech": ("Web Dev 101 – Udemy", "Master the foundations of HTML, CSS, and JavaScript.", "https://www.udemy.com/course/the-complete-web-development-bootcamp/"),
            "HR": ("HR Fundamentals – LinkedIn Learning", "Learn core HR responsibilities and hiring best practices.", "https://www.linkedin.com/learning/hr-fundamentals"),
        }
        if domain_selector in course_list:
            name_, desc, link = course_list[domain_selector]
            st.markdown(f"*[{name_}]({link})*  \n_{desc}_")
    else:
        st.markdown("You're already in great shape! Keep building your edge.")

    # Final Report (Mocked — actual generator in next part)
    if st.button("Download Full Report (PDF)"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "ResuGenius Report", ln=True, align="C")
        pdf.set_font("Arial", "", 12)
        pdf.ln(10)
        pdf.cell(0, 10, f"Candidate: {name}", ln=True)
        pdf.cell(0, 10, f"Domain: {domain_selector}", ln=True)
        pdf.cell(0, 10, f"Quiz Score: {quiz_score}/10", ln=True)
        pdf.cell(0, 10, f"Resume Match: {adjusted_score:.2f}", ln=True)
        pdf.ln(5)
        for trait, level in eq_profile.items():
            pdf.cell(0, 10, f"{trait}: {level}", ln=True)
        pdf.ln(5)
        pdf.cell(0, 10, "Report generated on " + datetime.today().strftime("%B %d, %Y"), ln=True)
        buffer = BytesIO()
        pdf.output(buffer)
        st.download_button("Download Report (PDF)", data=buffer.getvalue(), file_name=f"{name}_resugenius_report.pdf", mime="application/pdf")
