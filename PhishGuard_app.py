# phishing-email-detector
from pathlib import Path
import re

import joblib
import streamlit as st

from src.text_preprocessing import clean_text, get_stopwords


# -------------------------------------------------------------------
# Page setup
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.joblib"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"

st.set_page_config(
    page_title="PhishGuard | Email Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -------------------------------------------------------------------
# Load the saved model and preprocessing resources
# -------------------------------------------------------------------
@st.cache_resource
def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {VECTORIZER_PATH}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    stop_words = get_stopwords()
    return model, vectorizer, stop_words


# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --background: #08111f;
        --panel: #101d30;
        --panel-light: #14243a;
        --border: rgba(148, 163, 184, 0.17);
        --text: #edf4ff;
        --muted: #9aabc2;
        --accent: #42d6c5;
        --accent-dark: #123f45;
        --danger: #ff6b78;
        --success: #4adea6;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 12% 0%, rgba(32, 110, 130, .19), transparent 32%),
            radial-gradient(circle at 90% 12%, rgba(49, 91, 150, .18), transparent 28%),
            var(--background);
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(8, 17, 31, .75);
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--text) !important;
    }

    p, label, span {
        color: inherit;
    }

    .hero {
        padding: 2rem 0 1.5rem 0;
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 12px;
        border: 1px solid rgba(66, 214, 197, .28);
        border-radius: 999px;
        color: #8cece0;
        background: rgba(66, 214, 197, .08);
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.3rem, 5vw, 4rem);
        font-weight: 700;
        line-height: 1.08;
        letter-spacing: -.045em;
        margin: 1.1rem 0 .8rem 0;
        color: var(--text);
    }

    .hero-title span {
        color: var(--accent);
    }

    .hero-copy {
        max-width: 650px;
        color: var(--muted);
        font-size: 1.05rem;
        line-height: 1.7;
    }

    .panel {
        background: linear-gradient(145deg, rgba(20, 36, 58, .96), rgba(13, 27, 45, .96));
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.35rem;
        box-shadow: 0 18px 55px rgba(0, 0, 0, .16);
    }

    .panel-label {
        color: #a9bbd1;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .09em;
        text-transform: uppercase;
        margin-bottom: .45rem;
    }

    .panel-heading {
        color: var(--text);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: .35rem;
    }

    .panel-description {
        color: var(--muted);
        font-size: .9rem;
        line-height: 1.55;
        margin-bottom: 1rem;
    }

    .stTextArea textarea {
        background: #0a1728 !important;
        color: #edf4ff !important;
        border: 1px solid rgba(148, 163, 184, .25) !important;
        border-radius: 13px !important;
        line-height: 1.6 !important;
    }

    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }

    .stButton > button {
        border-radius: 11px;
        min-height: 2.8rem;
        font-weight: 700;
        border: 1px solid rgba(66, 214, 197, .45);
        background: linear-gradient(90deg, #32bfae, #42d6c5);
        color: #062126;
        transition: transform .15s ease, filter .15s ease;
    }

    .stButton > button:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
        border-color: #8cece0;
        color: #062126;
    }

    .stButton > button:focus {
        box-shadow: 0 0 0 2px rgba(66, 214, 197, .25);
    }

    .result-card {
        border-radius: 18px;
        padding: 1.35rem;
        margin-top: 1rem;
        border: 1px solid var(--border);
        background: rgba(16, 29, 48, .92);
    }

    .result-phishing {
        border-color: rgba(255, 107, 120, .45);
        background: linear-gradient(135deg, rgba(105, 35, 52, .30), rgba(16, 29, 48, .95));
    }

    .result-safe {
        border-color: rgba(74, 222, 166, .4);
        background: linear-gradient(135deg, rgba(24, 101, 78, .25), rgba(16, 29, 48, .95));
    }

    .result-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.55rem;
        font-weight: 700;
        margin: .25rem 0;
        color: var(--text);
    }

    .muted {
        color: var(--muted);
    }

    .stat {
        background: rgba(10, 23, 40, .8);
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: .9rem 1rem;
        min-height: 88px;
    }

    .stat-value {
        color: var(--text);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .stat-name {
        color: var(--muted);
        font-size: .78rem;
        margin-top: .15rem;
    }

    .footer {
        color: #71839b;
        text-align: center;
        font-size: .8rem;
        padding-top: 2.4rem;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">🛡️ Email threat protection</div>
        <div class="hero-title">Think before you <span>click.</span></div>
        <div class="hero-copy">
            Analyze suspicious emails in seconds. PhishGuard checks the
            message using a machine-learning model and highlights the
            potential risk before you interact with it.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Load model resources
# -------------------------------------------------------------------
try:
    model, vectorizer, stop_words = load_resources()
except Exception as exc:
    st.error(
        "PhishGuard could not load its model files. "
        "Check that both saved files are present in the project's models folder."
    )
    st.code(str(exc))
    st.stop()


# -------------------------------------------------------------------
# Input and examples
# -------------------------------------------------------------------
if "email_text" not in st.session_state:
    st.session_state.email_text = ""

left, right = st.columns([1.55, 1], gap="large")

with left:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-label">Email scanner</div>
            <div class="panel-heading">Paste an email to inspect</div>
            <div class="panel-description">
                Include the subject and full message body for a more useful analysis.
                Do not paste passwords, account credentials, or other private information.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    email_text = st.text_area(
        "Email content",
        key="email_text",
        height=265,
        placeholder=(
            "Paste the email subject and message here...\n\n"
            "Example: Your account has been suspended. Click the link below "
            "to verify your details immediately."
        ),
        label_visibility="collapsed",
    )

    analyze_col, clear_col = st.columns([3, 1])
    with analyze_col:
        analyze_clicked = st.button(
            "🔍  Analyze email",
            type="primary",
            use_container_width=True,
        )
    with clear_col:
        if st.button("Clear", use_container_width=True):
            st.session_state.email_text = ""
            st.session_state.pop("analysis_result", None)
            st.rerun()

with right:
    st.markdown(
        """
        <div class="panel">
            <div class="panel-label">Quick test</div>
            <div class="panel-heading">Try a sample email</div>
            <div class="panel-description">
                Select an example to see how the scanner responds.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    samples = {
        "Suspicious account alert": (
            "URGENT: Your bank account has been temporarily suspended. "
            "Verify your account immediately at http://secure-account-check.example "
            "or your access will be permanently blocked."
        ),
        "Prize notification": (
            "Congratulations! You have won a free reward. "
            "Click www.claim-your-prize.example now and submit your personal "
            "details to receive your prize."
        ),
        "Normal work email": (
            "Hi team, please find the meeting notes attached. "
            "Our next project review is scheduled for Thursday at 10 AM. "
            "Let me know if you have any questions. Thanks."
        ),
    }

    for sample_name, sample_text in samples.items():
        if st.button(sample_name, key=f"sample_{sample_name}", use_container_width=True):
            st.session_state.email_text = sample_text
            st.session_state.pop("analysis_result", None)
            st.rerun()

    st.markdown(
        """
        <div style="padding: 1rem 0 .2rem 0; color: #9aabc2; font-size: .84rem; line-height: 1.6;">
            <b style="color:#dce8f7;">Safety reminder</b><br>
            This tool provides an automated assessment, not a guarantee.
            Avoid clicking links or opening attachments in messages you do not trust.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# Prediction helpers
# -------------------------------------------------------------------
def is_phishing_label(label):
    """Interpret the saved model's label using the project's 1 = phishing convention."""
    value = str(label).strip().lower()

    if value in {"phishing", "phish", "spam", "malicious", "1", "1.0", "true"}:
        return True
    if value in {"legitimate", "legit", "safe", "ham", "0", "0.0", "false"}:
        return False

    # For numeric binary labels, retain the convention used by the original app.
    try:
        return int(float(value)) == 1
    except (TypeError, ValueError):
        return False


def analyze_email(raw_email):
    cleaned_email = clean_text(raw_email, stop_words)
    if not cleaned_email.strip():
        raise ValueError(
            "The email does not contain enough readable text to analyze. "
            "Try pasting the message again."
        )

    features = vectorizer.transform([cleaned_email])
    prediction = model.predict(features)[0]
    phishing = is_phishing_label(prediction)

    phishing_probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = list(model.classes_)
        phishing_indices = [
            index for index, label in enumerate(classes)
            if is_phishing_label(label)
        ]
        if phishing_indices:
            phishing_probability = float(probabilities[phishing_indices[0]]) * 100

    urls = re.findall(r"(?:https?://|www\.)\S+", raw_email, flags=re.IGNORECASE)
    words = re.findall(r"\b[\w']+\b", raw_email)
    urgency_terms = [
        "urgent", "immediately", "verify", "suspended", "expire",
        "limited time", "act now", "confirm", "password", "blocked",
    ]
    lower_email = raw_email.lower()
    detected_urgency = [term for term in urgency_terms if term in lower_email]

    return {
        "phishing": phishing,
        "probability": phishing_probability,
        "word_count": len(words),
        "character_count": len(raw_email),
        "url_count": len(urls),
        "urgency_terms": detected_urgency,
    }


# -------------------------------------------------------------------
# Run analysis and show result
# -------------------------------------------------------------------
if analyze_clicked:
    if not email_text.strip():
        st.warning("Please paste an email before clicking Analyze.")
    else:
        try:
            st.session_state.analysis_result = analyze_email(email_text)
        except Exception as exc:
            st.session_state.pop("analysis_result", None)
            st.error(str(exc))

result = st.session_state.get("analysis_result")

if result:
    st.markdown("---")
    st.markdown("## Analysis result")

    if result["phishing"]:
        result_class = "result-phishing"
        icon = "🚨"
        heading = "Potential phishing email"
        description = (
            "The model identified patterns associated with phishing. "
            "Do not click links, open attachments, or share personal information."
        )
        status_color = "#ff8791"
    else:
        result_class = "result-safe"
        icon = "🛡️"
        heading = "Likely legitimate email"
        description = (
            "The model did not classify this message as phishing. "
            "Still verify the sender before trusting links or attachments."
        )
        status_color = "#65e6b1"

    st.markdown(
        f"""
        <div class="result-card {result_class}">
            <div style="font-size:2rem;">{icon}</div>
            <div class="result-title" style="color:{status_color};">{heading}</div>
            <div class="muted">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if result["probability"] is not None:
        probability = result["probability"]
        st.markdown("### Phishing probability")
        st.progress(max(0.0, min(1.0, probability / 100)))
        st.markdown(
            f"<div style='color:#dce8f7;font-weight:700;'>{probability:.1f}%</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Email overview")
    stat1, stat2, stat3, stat4 = st.columns(4)
    stats = [
        (result["word_count"], "Words"),
        (result["character_count"], "Characters"),
        (result["url_count"], "Links detected"),
        (len(result["urgency_terms"]), "Urgency signals"),
    ]

    for column, (value, label) in zip([stat1, stat2, stat3, stat4], stats):
        with column:
            st.markdown(
                f"""
                <div class="stat">
                    <div class="stat-value">{value}</div>
                    <div class="stat-name">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if result["urgency_terms"]:
        st.markdown("### Words that may need attention")
        st.caption(", ".join(result["urgency_terms"]))
    else:
        st.markdown("### Message signals")
        st.caption("No common urgency phrases were detected by the simple keyword check.")


# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        PHISHGUARD · Machine-learning email screening<br>
        Use this result as a warning signal, not as a substitute for careful verification.
    </div>
    """,
    unsafe_allow_html=True,
)
