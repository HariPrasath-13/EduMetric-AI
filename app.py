import os
import sys
import time
import json
import random
import datetime
import importlib
import pandas as pd
import streamlit as st
from PIL import Image

# Add current directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import models.model_pipeline
import utils.evaluation
import utils.auth_service
import utils.email_service

importlib.reload(models.model_pipeline)
importlib.reload(utils.evaluation)
importlib.reload(utils.auth_service)
importlib.reload(utils.email_service)

from models.model_pipeline import (
    extract_text_from_pdf,
    clean_text,
    page_aware_semantic_chunking,
    DocumentVectorIndex,
    run_all_four_models
)
from utils.evaluation import (
    MODEL_NAMES,
    MODEL_COLORS,
    sanitize_filename,
    save_complete_assessment_bundle,
    build_authoritative_metrics_df,
    validate_metrics,
    diagnose_student_weak_areas
)
from utils.auth_service import (
    register_student,
    authenticate_student,
    find_user_by_identifier,
    add_student_assessment_record,
    get_student_assessment_history,
    load_all_users
)
from utils.email_service import (
    generate_secure_otp,
    send_otp_via_email,
    mask_email,
    get_smtp_config,
    save_smtp_config
)

# ============================================================================
# PAGE CONFIGURATION & LIGHT-THEME STYLING
# ============================================================================

st.set_page_config(
    page_title="EduMetric AI | Student Assessment & Evaluation Portal",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Light Mode CSS (Strict Light Theme Enforcement)
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Force overall light theme on all Streamlit containers */
html, body, [data-testid="stAppViewContainer"], .stApp, section[data-testid="stSidebar"], .main {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #1E293B !important;
    background-color: #F8FAFC !important;
}

header[data-testid="stHeader"] {
    background-color: #F8FAFC !important;
}

/* Force light background on main block */
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1400px;
    background-color: #F8FAFC !important;
}

/* Tabs styling */
button[data-baseweb="tab"] {
    color: #64748B !important;
    font-weight: 600 !important;
    background-color: transparent !important;
    font-size: 0.95rem !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #4F46E5 !important;
    border-bottom: 2px solid #4F46E5 !important;
    font-weight: 700 !important;
}

/* Card aesthetics */
.edu-card {
    background-color: #FFFFFF !important;
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 14px !important;
    padding: 22px 26px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    margin-bottom: 18px !important;
    color: #1E293B !important;
}

.edu-card h1, .edu-card h2, .edu-card h3, .edu-card h4, .edu-card p, .edu-card span {
    color: #1E293B !important;
}

.edu-header-card {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    color: #FFFFFF !important;
    border-radius: 14px !important;
    padding: 24px 28px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.18) !important;
}

.edu-header-card h1, .edu-header-card h2, .edu-header-card h3, .edu-header-card p, .edu-header-card span {
    color: #FFFFFF !important;
    margin: 0;
}

.student-hero-card {
    background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%) !important;
    color: #FFFFFF !important;
    border-radius: 16px !important;
    padding: 26px 30px !important;
    margin-bottom: 22px !important;
    box-shadow: 0 8px 24px rgba(30, 27, 75, 0.18) !important;
}

.student-hero-card h1, .student-hero-card h2, .student-hero-card h3, .student-hero-card p {
    color: #FFFFFF !important;
}

.student-hero-card span:not(.student-id-pill) {
    color: #FFFFFF !important;
}

/* Metric badges */
.metric-pill {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.85rem;
    background-color: #EEF2FF !important;
    color: #4F46E5 !important;
    border: 1px solid #C7D2FE !important;
}

.student-id-pill {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    padding: 6px 16px !important;
    border-radius: 20px !important;
    font-weight: 800 !important;
    font-size: 0.88rem !important;
    background-color: #FFFFFF !important;
    color: #1E1B4B !important;
    border: 1.5px solid #C7D2FE !important;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12) !important;
}

.student-hero-card .student-id-pill,
.student-hero-card .student-id-pill * {
    color: #1E1B4B !important;
    background-color: #FFFFFF !important;
}

/* Status colors */
.badge-success {
    background-color: #ECFDF5 !important;
    color: #059669 !important;
    border: 1px solid #A7F3D0 !important;
    padding: 4px 12px !important;
    border-radius: 12px !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
}

.badge-warning {
    background-color: #FFFBEB !important;
    color: #D97706 !important;
    border: 1px solid #FDE68A !important;
    padding: 4px 12px !important;
    border-radius: 12px !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
}

.badge-info {
    background-color: #EEF2FF !important;
    color: #4F46E5 !important;
    border: 1px solid #C7D2FE !important;
    padding: 4px 12px !important;
    border-radius: 12px !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
}

/* Radio button option cards */
div[role="radiogroup"] > label {
    background-color: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 10px !important;
    padding: 12px 18px !important;
    margin-bottom: 10px !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    display: flex !important;
    align-items: center !important;
}

div[role="radiogroup"] > label div, div[role="radiogroup"] > label span, div[role="radiogroup"] > label p {
    color: #0F172A !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
}

div[role="radiogroup"] > label:hover {
    border-color: #4F46E5 !important;
    background-color: #F5F3FF !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    padding: 0.5rem 1.2rem;
    transition: all 0.2s ease;
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
    color: #FFFFFF !important;
    border: none;
    box-shadow: 0 2px 8px rgba(79, 70, 229, 0.25);
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #4338CA 0%, #4F46E5 100%) !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.35);
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #F1F5F9;
}
::-webkit-scrollbar-thumb {
    background: #CBD5E1;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #94A3B8;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================================
# SESSION STATE INITIALIZATION & LIFECYCLE
# ============================================================================

def init_session():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "otp_verified" not in st.session_state:
        st.session_state.otp_verified = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "otp_dispatched" not in st.session_state:
        st.session_state.otp_dispatched = False
    if "pending_auth_user" not in st.session_state:
        st.session_state.pending_auth_user = None
    if "pending_otp" not in st.session_state:
        st.session_state.pending_otp = None
    if "otp_timestamp" not in st.session_state:
        st.session_state.otp_timestamp = None
    if "otp_email_status" not in st.session_state:
        st.session_state.otp_email_status = None
    if "current_nav" not in st.session_state:
        st.session_state.current_nav = "Dashboard"
    if "assessment_id" not in st.session_state:
        st.session_state.assessment_id = None
    if "generated_data" not in st.session_state:
        st.session_state.generated_data = None
    if "assessment_status" not in st.session_state:
        st.session_state.assessment_status = "SETUP"  # SETUP, GENERATING, IN_PROGRESS, COMPLETED
    if "student_answers" not in st.session_state:
        st.session_state.student_answers = {}
    if "current_q_idx" not in st.session_state:
        st.session_state.current_q_idx = 0
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "submitted_results" not in st.session_state:
        st.session_state.submitted_results = None
    if "generation_summaries" not in st.session_state:
        st.session_state.generation_summaries = None


def reset_to_new_assessment():
    """Resets only assessment-specific session state while preserving student login."""
    st.session_state.assessment_id = None
    st.session_state.assessment_status = "SETUP"
    st.session_state.generated_data = None
    st.session_state.student_answers = {}
    st.session_state.current_q_idx = 0
    st.session_state.start_time = None
    st.session_state.submitted_results = None
    st.session_state.generation_summaries = None
    st.session_state.current_nav = "Assessment"


init_session()


# ============================================================================
# LOGIN, REGISTRATION & SECURE 2-STEP OTP VERIFICATION
# ============================================================================

def render_login_screen():
    col1, col2, col3 = st.columns([1, 1.35, 1])
    with col2:
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='edu-header-card' style='text-align: center;'>
            <h2 style='font-size: 1.65rem;'>🎓 EduMetric Assessment Portal</h2>
            <p style='font-size: 0.92rem; opacity: 0.95; margin-top: 6px;'>
                Secure Multi-Model Assessment & RAG Evaluation Platform
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 1. OTP Verification Screen (Triggered after Sign In or Register)
        if st.session_state.otp_dispatched and st.session_state.pending_auth_user:
            user = st.session_state.pending_auth_user
            masked_mail = mask_email(user.get("email", ""))
            
            st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
            st.markdown("""
            <div style='text-align: center; margin-bottom: 16px;'>
                <div style='width: 54px; height: 54px; border-radius: 50%; background: #EEF2FF; color: #4F46E5; display: inline-flex; align-items: center; justify-content: center; font-size: 1.6rem; margin-bottom: 8px;'>
                    🔒
                </div>
                <h3 style='margin: 0; color: #1E293B;'>2-Step Email Verification</h3>
                <p style='font-size: 0.88rem; color: #64748B; margin-top: 4px;'>
                    A one-time verification password has been sent to your registered email.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style='background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 16px; margin-bottom: 18px; font-size: 0.88rem;'>
                <div style='margin-bottom: 4px;'><b>Student Name:</b> {user.get('name')}</div>
                <div style='margin-bottom: 4px;'><b>Unique Student ID:</b> <code style='color:#4F46E5; font-weight:700;'>{user.get('student_id')}</code></div>
                <div style='margin-bottom: 4px;'><b>Institution:</b> {user.get('institution')}</div>
                <div><b>Email Dispatch:</b> <span style='color:#059669; font-weight:600;'>{masked_mail}</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show email status notification
            status_info = st.session_state.otp_email_status or {}
            if status_info.get("sent_real_email"):
                st.success(f"📧 Live verification email sent to **{user.get('email')}**. Please check your Inbox and Spam folder.")
            else:
                st.warning("⚠️ **Outgoing Email Server (SMTP) is not configured yet.** Live emails require sender SMTP credentials.")
                
            # Quick SMTP Configuration Expander
            with st.expander("⚙️ Connect Your Gmail / Sender Email (Send Real Emails)", expanded=(not status_info.get("sent_real_email"))):
                st.markdown("""
                <div style='font-size:0.84rem; color:#475569; line-height:1.5; margin-bottom:10px;'>
                    To receive real emails in your inbox, enter your Gmail address and a 16-character <b>Google App Password</b> 
                    (Generate one in 30s at <a href='https://myaccount.google.com/apppasswords' target='_blank'>myaccount.google.com/apppasswords</a>).
                </div>
                """, unsafe_allow_html=True)
                cfg = get_smtp_config()
                q_host = st.text_input("SMTP Host", value=cfg.get("smtp_host", "smtp.gmail.com"), key="otp_smtp_host")
                q_port = st.number_input("SMTP Port", value=int(cfg.get("smtp_port", 587)), key="otp_smtp_port")
                q_user = st.text_input("Sender Gmail Address", value=cfg.get("smtp_user", ""), placeholder="your_gmail@gmail.com", key="otp_smtp_user")
                q_pwd = st.text_input("Gmail 16-char App Password", value=cfg.get("smtp_password", ""), type="password", placeholder="xxxx xxxx xxxx xxxx", key="otp_smtp_pwd")
                
                if st.button("💾 Save & Send Live OTP to My Email", type="secondary", use_container_width=True):
                    if q_user and q_pwd:
                        save_smtp_config(q_host, int(q_port), q_user, q_pwd, True)
                        res = send_otp_via_email(
                            recipient_email=user.get("email"),
                            recipient_name=user.get("name"),
                            otp_code=st.session_state.pending_otp,
                            student_id=user.get("student_id")
                        )
                        st.session_state.otp_email_status = res
                        if res.get("sent_real_email"):
                            st.success(f"✅ Email dispatched to {user.get('email')}!")
                        else:
                            st.error(f"❌ Could not send email: {res.get('message')}")
                        st.rerun()
                    else:
                        st.error("Please enter both sender Gmail address and App Password.")

            # Offline / Demo Helper Expander
            with st.expander("🔑 Offline / Instant Testing Helper", expanded=False):
                st.markdown(f"""
                <div style='font-size:0.86rem; color:#475569;'>
                    If you are testing locally without setting up Gmail SMTP, your active 6-digit code is:
                    <div style='font-size:1.4rem; font-weight:800; color:#4F46E5; margin:6px 0; font-family:monospace;'>
                        {st.session_state.pending_otp}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("✨ Auto-Fill Code into Box", key="autofill_btn"):
                    st.session_state.temp_otp_input = st.session_state.pending_otp
                    st.rerun()
                
            otp_val = st.session_state.get("temp_otp_input", "")
            otp_entered = st.text_input(
                "Enter 6-Digit OTP Code",
                value=otp_val,
                max_chars=6,
                placeholder="e.g. 849201",
                help="Enter the 6-digit verification code."
            )
            
            c_btn1, c_btn2 = st.columns([1.3, 1])
            with c_btn1:
                if st.button("✅ Verify & Enter Portal", type="primary", use_container_width=True):
                    # Check OTP validity
                    entered_clean = otp_entered.strip()
                    correct_otp = st.session_state.pending_otp
                    issued_time = st.session_state.otp_timestamp or 0
                    is_expired = (time.time() - issued_time) > 600  # 10 mins
                    
                    if is_expired:
                        st.error("Verification code has expired. Please click 'Resend Code'.")
                    elif entered_clean and entered_clean == correct_otp:
                        st.session_state.authenticated = True
                        st.session_state.otp_verified = True
                        st.session_state.current_user = st.session_state.pending_auth_user
                        st.session_state.current_nav = "Dashboard"
                        st.session_state.otp_dispatched = False
                        st.session_state.pending_auth_user = None
                        st.session_state.pending_otp = None
                        if "temp_otp_input" in st.session_state:
                            del st.session_state["temp_otp_input"]
                        st.success("Verification successful! Welcome.")
                        time.sleep(0.3)
                        st.rerun()
                    else:
                        st.error("Invalid verification code. Please check your email or resend.")
                        
            with c_btn2:
                if st.button("🔄 Resend Code", use_container_width=True):
                    new_otp = generate_secure_otp()
                    st.session_state.pending_otp = new_otp
                    st.session_state.otp_timestamp = time.time()
                    dispatch_res = send_otp_via_email(
                        recipient_email=user.get("email"),
                        recipient_name=user.get("name"),
                        otp_code=new_otp,
                        student_id=user.get("student_id")
                    )
                    st.session_state.otp_email_status = dispatch_res
                    st.info("A fresh verification code has been dispatched.")
                    st.rerun()
                    
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button("⬅️ Back to Sign In / Register", use_container_width=True):
                st.session_state.otp_dispatched = False
                st.session_state.pending_auth_user = None
                st.session_state.pending_otp = None
                if "temp_otp_input" in st.session_state:
                    del st.session_state["temp_otp_input"]
                st.rerun()
                
            st.markdown("</div>", unsafe_allow_html=True)
            return

        # 2. Main Authentication Box: Sign In & Register Tabs
        st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
        tab_signin, tab_register = st.tabs(["🔑 Sign In", "📝 Register New Student"])
        
        # --- TAB 1: SIGN IN ---
        with tab_signin:
            st.markdown("<p style='font-size:0.88rem; color:#64748B; margin-top:4px;'>Sign in with your registered Email or Unique Student ID.</p>", unsafe_allow_html=True)
            
            login_ident = st.text_input(
                "Email Address / Student ID",
                placeholder="e.g. student@university.edu or STU-2026-8819",
                key="login_ident_input"
            )
            login_pwd = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="login_pwd_input"
            )
            
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button("🚀 Sign In & Receive OTP via Email", type="primary", use_container_width=True):
                if not login_ident or not login_pwd:
                    st.error("Please enter both Email/Student ID and Password.")
                else:
                    ok, msg, user_obj = authenticate_student(login_ident, login_pwd)
                    if ok and user_obj:
                        otp_code = generate_secure_otp()
                        st.session_state.pending_auth_user = user_obj
                        st.session_state.pending_otp = otp_code
                        st.session_state.otp_timestamp = time.time()
                        st.session_state.otp_dispatched = True
                        
                        email_res = send_otp_via_email(
                            recipient_email=user_obj.get("email"),
                            recipient_name=user_obj.get("name"),
                            otp_code=otp_code,
                            student_id=user_obj.get("student_id")
                        )
                        st.session_state.otp_email_status = email_res
                        st.rerun()
                    else:
                        st.error(msg)
                        
        # --- TAB 2: REGISTER ---
        with tab_register:
            st.markdown("<p style='font-size:0.88rem; color:#64748B; margin-top:4px;'>Create your account. A <b>unique Student ID</b> will be automatically assigned to you.</p>", unsafe_allow_html=True)
            
            reg_name = st.text_input("Full Name *", placeholder="e.g. Hariprasath R", key="reg_name_input")
            
            c_inst1, c_inst2 = st.columns(2)
            with c_inst1:
                reg_institution = st.text_input("College or School Name *", placeholder="e.g. Sri Eshwar College of Engineering", key="reg_inst_input")
            with c_inst2:
                reg_dept = st.text_input("Department / Branch", placeholder="e.g. Computer Science & Eng.", key="reg_dept_input")
                
            reg_email = st.text_input("Email Address (for OTP Verification) *", placeholder="e.g. hariprasath@gmail.com", key="reg_email_input")
            
            c_pwd1, c_pwd2 = st.columns(2)
            with c_pwd1:
                reg_pwd = st.text_input("Password (min 6 chars) *", type="password", key="reg_pwd_input")
            with c_pwd2:
                reg_pwd_confirm = st.text_input("Confirm Password *", type="password", key="reg_pwd_conf_input")
                
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button("🎓 Register & Generate Student ID", type="primary", use_container_width=True):
                if not reg_name.strip():
                    st.error("Please enter your Full Name.")
                elif not reg_institution.strip():
                    st.error("Please enter your College or School name.")
                elif not reg_email.strip() or "@" not in reg_email:
                    st.error("Please provide a valid Email Address.")
                elif len(reg_pwd) < 6:
                    st.error("Password must be at least 6 characters long.")
                elif reg_pwd != reg_pwd_confirm:
                    st.error("Passwords do not match. Please re-enter.")
                else:
                    ok, msg, new_user = register_student(
                        name=reg_name,
                        institution=reg_institution,
                        email=reg_email,
                        password=reg_pwd,
                        department=reg_dept or "Computer Science & Engineering"
                    )
                    if ok and new_user:
                        otp_code = generate_secure_otp()
                        st.session_state.pending_auth_user = new_user
                        st.session_state.pending_otp = otp_code
                        st.session_state.otp_timestamp = time.time()
                        st.session_state.otp_dispatched = True
                        
                        email_res = send_otp_via_email(
                            recipient_email=new_user.get("email"),
                            recipient_name=new_user.get("name"),
                            otp_code=otp_code,
                            student_id=new_user.get("student_id")
                        )
                        st.session_state.otp_email_status = email_res
                        st.success(f"Account registered! Your Unique Student ID is: {new_user.get('student_id')}")
                        time.sleep(0.3)
                        st.rerun()
                    else:
                        st.error(msg)
                        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Optional Collapsible SMTP Configuration Helper for Real Email Testing
        with st.expander("⚙️ Email & SMTP Server Settings (Optional)", expanded=False):
            st.markdown("<p style='font-size:0.82rem; color:#64748B;'>Configure custom SMTP settings (e.g. Gmail App Password) to send live OTP emails directly to student mailboxes.</p>", unsafe_allow_html=True)
            cfg = get_smtp_config()
            cfg_host = st.text_input("SMTP Host", value=cfg.get("smtp_host", "smtp.gmail.com"), key="smtp_host_cfg")
            cfg_port = st.number_input("SMTP Port", value=int(cfg.get("smtp_port", 587)), key="smtp_port_cfg")
            cfg_user = st.text_input("Sender Email / User", value=cfg.get("smtp_user", ""), placeholder="e.g. your_email@gmail.com", key="smtp_user_cfg")
            cfg_pwd = st.text_input("SMTP / Gmail App Password", value=cfg.get("smtp_password", ""), type="password", placeholder="16-character App Password", key="smtp_pwd_cfg")
            cfg_tls = st.checkbox("Use STARTTLS (Port 587)", value=cfg.get("smtp_use_tls", True), key="smtp_tls_cfg")
            
            if st.button("Save SMTP Settings"):
                if save_smtp_config(cfg_host, int(cfg_port), cfg_user, cfg_pwd, cfg_tls):
                    st.success("SMTP configuration updated successfully!")


if not (st.session_state.authenticated and st.session_state.otp_verified and st.session_state.current_user):
    render_login_screen()
    st.stop()


# ============================================================================
# SIDEBAR NAVIGATION & STUDENT PROFILE
# ============================================================================

current_student = st.session_state.current_user or {}
student_name = current_student.get("name", "Student")
student_id = current_student.get("student_id", "STU-2026-8819")
student_institution = current_student.get("institution", "Sri Eshwar College of Engineering")

# Extract initials
name_parts = student_name.split()
initials = (name_parts[0][0] + (name_parts[1][0] if len(name_parts) > 1 else "")).upper() if name_parts else "ST"

with st.sidebar:
    st.markdown(f"""
    <div style='text-align: center; padding: 12px 0 16px 0; border-bottom: 1px solid #E2E8F0;'>
        <div style='width: 54px; height: 54px; border-radius: 50%; background: linear-gradient(135deg, #4F46E5, #7C3AED); color: white; display: flex; align-items: center; justify-content: center; font-size: 1.35rem; font-weight: 800; margin: 0 auto 10px auto; box-shadow: 0 4px 10px rgba(79, 70, 229, 0.25);'>
            {initials}
        </div>
        <h4 style='margin: 0; font-size: 1.05rem; color: #1E293B; font-weight: 700;'>{student_name}</h4>
        <p style='margin: 3px 0 0 0; font-size: 0.8rem; color: #4F46E5; font-weight: 800;'>🆔 {student_id}</p>
        <p style='margin: 2px 0 0 0; font-size: 0.78rem; color: #64748B;'>🏫 {student_institution}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 0.75rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-top: 14px; letter-spacing: 0.5px;'>Navigation</p>", unsafe_allow_html=True)
    nav_selection = st.radio(
        "Menu",
        ["Dashboard", "Assessment", "Performance", "Model Comparison"],
        index=["Dashboard", "Assessment", "Performance", "Model Comparison"].index(st.session_state.current_nav) if st.session_state.current_nav in ["Dashboard", "Assessment", "Performance", "Model Comparison"] else 0,
        label_visibility="collapsed"
    )
    st.session_state.current_nav = nav_selection
    
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    if st.button("🚪 Log Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.otp_verified = False
        st.session_state.current_user = None
        st.session_state.otp_dispatched = False
        st.session_state.pending_auth_user = None
        st.session_state.pending_otp = None
        st.session_state.assessment_status = "SETUP"
        st.session_state.generated_data = None
        st.session_state.submitted_results = None
        st.rerun()


# ============================================================================
# VIEW 1: STUDENT DASHBOARD (STUDENT INFO & INTEGRATED HISTORY)
# ============================================================================

def render_dashboard_view():
    student = st.session_state.current_user or {}
    s_name = student.get("name", "Student")
    s_id = student.get("student_id", "STU-2026-8819")
    s_inst = student.get("institution", "Sri Eshwar College of Engineering")
    s_email = student.get("email", "student@university.edu")
    s_dept = student.get("department", "Computer Science & Engineering")
    
    name_parts = s_name.split()
    s_init = (name_parts[0][0] + (name_parts[1][0] if len(name_parts) > 1 else "")).upper() if name_parts else "ST"
    
    # 1. Student Hero Profile Banner
    st.markdown(f"""
    <div class='student-hero-card'>
        <div style='display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 16px;'>
            <div style='display: flex; align-items: center; gap: 18px;'>
                <div style='width: 68px; height: 68px; border-radius: 50%; background: #FFFFFF; color: #4338CA; display: flex; align-items: center; justify-content: center; font-size: 1.7rem; font-weight: 800; box-shadow: 0 4px 14px rgba(0,0,0,0.15);'>
                    {s_init}
                </div>
                <div>
                    <h2 style='margin: 0; font-size: 1.65rem; font-weight: 800; color: #FFFFFF;'>{s_name}</h2>
                    <p style='margin: 4px 0 0 0; font-size: 0.95rem; opacity: 0.92; color: #E0E7FF;'>
                        🏫 <b>{s_inst}</b> &nbsp;|&nbsp; 📚 {s_dept}
                    </p>
                    <div style='margin-top: 8px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap;'>
                        <span class='student-id-pill' style='display: inline-flex; align-items: center; background-color: #FFFFFF !important; color: #1E1B4B !important; font-weight: 800; padding: 6px 16px; border-radius: 20px; border: 1.5px solid #C7D2FE;'>
                            🆔 Student ID: <b style='color: #1E1B4B !important; margin-left: 4px;'>{s_id}</b>
                        </span>
                        <span style='font-size: 0.85rem; color: #C7D2FE;'>✉️ {s_email}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Fetch student history
    history_records = get_student_assessment_history(s_id)
    total_tests = len(history_records)
    
    avg_pct = 0.0
    max_pct = 0.0
    total_q_solved = 0
    if total_tests > 0:
        pct_list = [h.get("percentage", 0.0) for h in history_records]
        avg_pct = sum(pct_list) / total_tests
        max_pct = max(pct_list)
        total_q_solved = sum([h.get("total", 0) for h in history_records])
        
    # 2. Student Performance Metric Summary Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='edu-card' style='text-align: center; border-top: 4px solid #4F46E5;'>
            <p style='font-size: 0.82rem; color: #64748B; margin: 0; font-weight: 600; text-transform: uppercase;'>Total Tests Taken</p>
            <h2 style='color: #4F46E5; margin: 8px 0 0 0; font-size: 1.85rem; font-weight: 800;'>{total_tests}</h2>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='edu-card' style='text-align: center; border-top: 4px solid #059669;'>
            <p style='font-size: 0.82rem; color: #64748B; margin: 0; font-weight: 600; text-transform: uppercase;'>Average Score</p>
            <h2 style='color: #059669; margin: 8px 0 0 0; font-size: 1.85rem; font-weight: 800;'>{avg_pct:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='edu-card' style='text-align: center; border-top: 4px solid #D97706;'>
            <p style='font-size: 0.82rem; color: #64748B; margin: 0; font-weight: 600; text-transform: uppercase;'>Highest Score</p>
            <h2 style='color: #D97706; margin: 8px 0 0 0; font-size: 1.85rem; font-weight: 800;'>{max_pct:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='edu-card' style='text-align: center; border-top: 4px solid #0284C7;'>
            <p style='font-size: 0.82rem; color: #64748B; margin: 0; font-weight: 600; text-transform: uppercase;'>Questions Solved</p>
            <h2 style='color: #0284C7; margin: 8px 0 0 0; font-size: 1.85rem; font-weight: 800;'>{total_q_solved}</h2>
        </div>
        """, unsafe_allow_html=True)
        
    # Quick Start Action Banner
    col_act1, col_act2 = st.columns([3, 1])
    with col_act1:
        st.markdown("""
        <div style='padding: 6px 0;'>
            <h3 style='margin: 0; font-size: 1.3rem; color: #1E293B;'>📜 Assessment & Test History</h3>
            <p style='margin: 2px 0 0 0; font-size: 0.88rem; color: #64748B;'>
                Comprehensive log of your PDF-grounded evaluations, test scores, and 4-model comparative benchmarks.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_act2:
        if st.button("🚀 Start New Assessment", type="primary", use_container_width=True):
            reset_to_new_assessment()
            st.session_state.current_nav = "Assessment"
            st.rerun()
            
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    
    # 3. Assessment History Records List & Drill-down
    if not history_records:
        st.markdown("""
        <div class='edu-card' style='text-align: center; padding: 40px 20px;'>
            <div style='font-size: 2.5rem; margin-bottom: 10px;'>📚</div>
            <h3 style='color: #1E293B; margin: 0;'>No Assessment History Yet</h3>
            <p style='color: #64748B; font-size: 0.92rem; max-width: 480px; margin: 8px auto 18px auto;'>
                You haven't completed any assessments yet. Upload a syllabus or textbook PDF to test your knowledge and compare AI generation models!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📝 Upload PDF & Take First Assessment", type="primary"):
            reset_to_new_assessment()
            st.session_state.current_nav = "Assessment"
            st.rerun()
            return
            
    # Search / Filter Bar for History
    search_q = st.text_input("🔍 Filter assessment history by document or topic name...", placeholder="e.g. Computer Networks or Unit 1", label_visibility="collapsed")
    
    filtered_history = [
        h for h in history_records 
        if not search_q or search_q.lower() in h.get("pdf_name", "").lower() or search_q.lower() in h.get("rel_name", "").lower()
    ]
    
    # History Table Overview
    table_data = []
    for idx, h in enumerate(filtered_history):
        pct_val = h.get("percentage", 0.0)
        if pct_val >= 80:
            status_str = "🌟 Distinction"
        elif pct_val >= 60:
            status_str = "👍 Passed"
        else:
            status_str = "⚠️ Needs Review"
            
        table_data.append({
            "#": idx + 1,
            "Date & Time": h.get("submitted_at", "N/A"),
            "Study Material (PDF)": h.get("pdf_name", "N/A"),
            "Score": f"{h.get('score', 0)} / {h.get('total', 0)}",
            "Accuracy": f"{pct_val:.1f}%",
            "Time Taken": h.get("time_taken", "N/A"),
            "Performance": status_str
        })
        
    df_history_view = pd.DataFrame(table_data)
    st.dataframe(df_history_view, use_container_width=True, hide_index=True)
    
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.subheader("📁 Detailed Test Run Reports & Artifacts")
    
    # Expandable detail cards for each history record
    for idx, h in enumerate(filtered_history):
        run_name = h.get("pdf_name", "Assessment Run")
        sub_time = h.get("submitted_at", "N/A")
        pct = h.get("percentage", 0.0)
        score_info = f"{h.get('score')}/{h.get('total')} ({pct:.1f}%)"
        
        expander_title = f"📄 [{idx+1}] {run_name} — Score: {score_info} | {sub_time}"
        with st.expander(expander_title, expanded=(idx == 0 and len(filtered_history) <= 3)):
            col_d1, col_d2 = st.columns([1.5, 1])
            with col_d1:
                st.markdown(f"**Document Name:** `{run_name}`")
                st.markdown(f"**Assessment ID:** `{h.get('assessment_id')}`")
                st.markdown(f"**Submitted At:** {sub_time} | **Time Taken:** {h.get('time_taken')}")
                st.markdown(f"**Student Score:** **{h.get('score')} / {h.get('total')}** ({pct:.1f}%)")
                
                # Show comparative metrics if available
                m_path = h.get("metrics_json_path")
                if m_path and os.path.exists(m_path):
                    try:
                        with open(m_path, "r", encoding="utf-8") as f:
                            m_data = json.load(f)
                        df_m = build_authoritative_metrics_df(m_data)
                        st.markdown("##### 🔬 4-Model Comparative Metrics for this Run:")
                        st.dataframe(df_m[["Model", "Accuracy", "Precision", "Recall", "F1", "Total Questions"]], use_container_width=True, hide_index=True)
                    except Exception:
                        pass
                        
            with col_d2:
                st.markdown("##### 📥 Run Downloads & Artifacts")
                # PDF report download
                rep_pdf = h.get("report_pdf_path")
                if rep_pdf and os.path.exists(rep_pdf):
                    with open(rep_pdf, "rb") as f:
                        st.download_button(
                            label="📄 Download Experiment Report PDF",
                            data=f.read(),
                            file_name=f"report_{sanitize_filename(run_name)}.pdf",
                            mime="application/pdf",
                            key=f"dl_pdf_dash_{idx}_{h.get('assessment_id')}",
                            use_container_width=True
                        )
                
                # Excel report download
                xl_p = h.get("excel_path")
                if xl_p and os.path.exists(xl_p):
                    with open(xl_p, "rb") as f:
                        st.download_button(
                            label="📊 Download Excel Metrics Workbook (.xlsx)",
                            data=f.read(),
                            file_name=os.path.basename(xl_p),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"dl_xl_dash_{idx}_{h.get('assessment_id')}",
                            use_container_width=True
                        )
                        
                # Metrics JSON download
                if m_path and os.path.exists(m_path):
                    with open(m_path, "rb") as f:
                        st.download_button(
                            label="📦 Download Metrics JSON",
                            data=f.read(),
                            file_name="metrics.json",
                            mime="application/json",
                            key=f"dl_j_dash_{idx}_{h.get('assessment_id')}",
                            use_container_width=True
                        )


# ============================================================================
# VIEW 2: ASSESSMENT (MAIN FLOW)
# ============================================================================

def render_assessment_view():
    # SUB-VIEW A: Configuration & PDF Upload (SETUP)
    if st.session_state.assessment_status in ["SETUP", "idle"]:
        st.markdown("""
        <div class='edu-header-card'>
            <h2>📝 PDF Study Material Assessment Setup</h2>
            <p style='margin-top: 4px; opacity: 0.9;'>Configure your assessment parameters and upload your study PDF</p>
        </div>
        """, unsafe_allow_html=True)
        
        col_up, col_cfg = st.columns([1.1, 1.2])
        
        with col_up:
            st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
            st.subheader("1. Upload Study Material (PDF)")
            uploaded_pdf = st.file_uploader("Select or Drag & Drop PDF", type=["pdf"], key="pdf_file_uploader")
            
            if uploaded_pdf is not None:
                st.success(f"📄 **Loaded:** `{uploaded_pdf.name}` ({round(len(uploaded_pdf.getvalue())/1024, 1)} KB)")
            else:
                st.info("ℹ️ Please upload a course syllabus, textbook chapter, or lecture notes PDF.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_cfg:
            st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
            st.subheader("2. Assessment Parameters")
            
            c_q, c_t = st.columns(2)
            with c_q:
                num_q = st.selectbox("Number of Questions", [5, 10, 15, 20, 25, 30], index=1)
            with c_t:
                time_lim = st.selectbox("Time Limit", ["5 minutes", "10 minutes", "15 minutes", "20 minutes", "30 minutes", "45 minutes", "60 minutes"], index=2)
                time_mins = int(time_lim.split()[0])
                
            difficulty = st.selectbox("Difficulty Distribution", ["Mixed", "Easy", "Medium", "Hard"], index=0)
            st.markdown("</div>", unsafe_allow_html=True)
            
        if uploaded_pdf is not None:
            if st.button("🚀 Start Assessment", type="primary", use_container_width=True):
                # Generate unique assessment ID
                timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                rand_hex = os.urandom(2).hex()
                assessment_id = f"assessment_{timestamp_str}_{rand_hex}"
                st.session_state.assessment_id = assessment_id
                
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
                
                def update_progress(pct, text):
                    progress_placeholder.progress(pct)
                    status_placeholder.info(f"⏳ {text}")
                    
                cfg_dict = {
                    "num_questions": num_q,
                    "time_limit": time_mins,
                    "difficulty": difficulty
                }
                
                try:
                    pdf_bytes = uploaded_pdf.getvalue()
                    all_questions, all_metrics, full_pdf_text, pages_data, generation_summaries = run_all_four_models(
                        pdf_bytes_or_file=pdf_bytes,
                        num_questions=num_q,
                        time_limit=time_mins,
                        difficulty=difficulty,
                        config=cfg_dict,
                        progress_callback=update_progress
                    )
                    
                    st.session_state.generated_data = {
                        "pdf_name": uploaded_pdf.name,
                        "pdf_bytes": pdf_bytes,
                        "questions": all_questions,
                        "metrics": all_metrics,
                        "full_pdf_text": full_pdf_text,
                        "pages_data": pages_data,
                        "config": cfg_dict
                    }
                    st.session_state.generation_summaries = generation_summaries
                    st.session_state.student_answers = {}
                    st.session_state.current_q_idx = 0
                    st.session_state.start_time = time.time()
                    st.session_state.assessment_status = "IN_PROGRESS"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during assessment generation: {str(e)}")
                    st.session_state.assessment_status = "SETUP"

    # SUB-VIEW B: Active Blind Assessment
    elif st.session_state.assessment_status == "IN_PROGRESS":
        data = st.session_state.generated_data
        my_model_questions = data["questions"].get("My Model", [])
        total_q = len(my_model_questions)
        curr_idx = st.session_state.current_q_idx
        
        if total_q == 0:
            st.error("No questions were generated. Please try again.")
            if st.button("Reset"):
                reset_to_new_assessment()
                st.rerun()
            return
            
        current_q = my_model_questions[curr_idx]
        q_id = current_q.get("id", curr_idx + 1)
        
        # Header banner with timer and progress
        elapsed_sec = int(time.time() - st.session_state.start_time) if st.session_state.start_time else 0
        limit_sec = data["config"]["time_limit"] * 60
        rem_sec = max(0, limit_sec - elapsed_sec)
        
        c_head1, c_head2, c_head3 = st.columns([1.5, 1.2, 1])
        with c_head1:
            st.markdown(f"#### 📝 Active Assessment: {data['pdf_name']}")
        with c_head2:
            import streamlit.components.v1 as components
            components.html(
                f"""
                <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; justify-content: center; align-items: center; height: 100%; margin: 0; padding: 0;">
                    <div style="background-color: #EEF2FF; border: 1.5px solid #6366F1; border-radius: 20px; padding: 4px 14px; display: inline-flex; align-items: center; box-shadow: 0 1px 3px rgba(79, 70, 229, 0.1);">
                        <span style="font-size: 1rem; margin-right: 6px;">⏱️</span>
                        <span style="font-size: 0.84rem; font-weight: 700; color: #4338CA; margin-right: 6px;">Time Left:</span>
                        <span id="countdown_timer_val" style="font-size: 0.95rem; font-weight: 800; color: #4F46E5; font-family: monospace;">--:--</span>
                    </div>
                </div>
                <script>
                    (function() {{
                        var totalSec = {rem_sec};
                        var el = document.getElementById('countdown_timer_val');
                        function update() {{
                            if (totalSec <= 0) {{
                                el.innerText = "00:00 (Time Up)";
                                el.style.color = "#DC2626";
                                return;
                            }}
                            var m = Math.floor(totalSec / 60);
                            var s = totalSec % 60;
                            el.innerText = (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
                            if (totalSec <= 60) {{
                                el.style.color = "#DC2626";
                            }}
                            totalSec--;
                        }}
                        update();
                        setInterval(update, 1000);
                    }})();
                </script>
                """,
                height=45
            )
        with c_head3:
            answered_count = len([k for k, v in st.session_state.student_answers.items() if v is not None])
            st.markdown(f"<div style='text-align:right; margin-top:8px;'><span class='badge-success'>Answered: {answered_count} / {total_q}</span></div>", unsafe_allow_html=True)
            
        st.progress((curr_idx + 1) / total_q)
        
        # Main Question Card
        st.markdown("<div class='edu-card'>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#6366F1; font-weight:700; font-size:0.9rem;'>QUESTION {curr_idx + 1} OF {total_q}</p>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:#1E293B; margin-top:4px;'>{current_q.get('question')}</h4>", unsafe_allow_html=True)
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # Options
        options = current_q.get("options", {})
        opt_keys = ['a', 'b', 'c', 'd']
        current_selection = st.session_state.student_answers.get(str(q_id), None)
        
        radio_options = [f"{k.upper()}. {options.get(k, '')}" for k in opt_keys if k in options]
        default_index = None
        if current_selection in opt_keys:
            default_index = opt_keys.index(current_selection)
            
        selected_radio = st.radio(
            f"Select Option for Question {curr_idx + 1}",
            radio_options,
            index=default_index,
            key=f"radio_q_{curr_idx}",
            label_visibility="collapsed"
        )
        
        if selected_radio:
            chosen_key = selected_radio.split('.')[0].strip().lower()
            st.session_state.student_answers[str(q_id)] = chosen_key
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Navigation & Submit controls
        c_prev, c_pal, c_next, c_sub = st.columns([1, 2, 1, 1.2])
        with c_prev:
            if st.button("⬅️ Previous", disabled=(curr_idx == 0), use_container_width=True):
                st.session_state.current_q_idx = max(0, curr_idx - 1)
                st.rerun()
                
        with c_pal:
            # Multi-row Palette for up to 30 questions
            items_per_row = min(10, total_q)
            for row_start in range(0, total_q, items_per_row):
                row_end = min(row_start + items_per_row, total_q)
                row_cols = st.columns(row_end - row_start)
                for col_idx, q_i in enumerate(range(row_start, row_end)):
                    q_num = q_i + 1
                    target_qid = str(my_model_questions[q_i].get("id", q_i + 1))
                    is_ans = target_qid in st.session_state.student_answers and st.session_state.student_answers[target_qid] is not None
                    btn_label = f"✓{q_num}" if is_ans else f"{q_num}"
                    btn_type = "primary" if q_i == curr_idx else "secondary"
                    if row_cols[col_idx].button(btn_label, key=f"pal_{q_i}", type=btn_type, use_container_width=True):
                        st.session_state.current_q_idx = q_i
                        st.rerun()
                        
        with c_next:
            if st.button("Next ➡️", disabled=(curr_idx == total_q - 1), use_container_width=True):
                st.session_state.current_q_idx = min(total_q - 1, curr_idx + 1)
                st.rerun()
                
        with c_sub:
            if st.button("🏁 Submit Test", type="primary", use_container_width=True):
                # Calculate student results
                correct_count = 0
                for q in my_model_questions:
                    qid = str(q.get("id"))
                    user_ans = st.session_state.student_answers.get(qid, "")
                    if user_ans.lower() == str(q.get("correct_answer")).lower():
                        correct_count += 1
                        
                pct = (correct_count / total_q) * 100 if total_q > 0 else 0.0
                time_taken_str = f"{elapsed_sec // 60}m {elapsed_sec % 60}s"
                
                curr_user = st.session_state.get("current_user", {})
                student_results = {
                    "student_id": curr_user.get("student_id", "STU-2026-8819"),
                    "student_name": curr_user.get("name", "Student"),
                    "institution": curr_user.get("institution", "Institution"),
                    "score": correct_count,
                    "total": total_q,
                    "percentage": round(pct, 2),
                    "time_taken": time_taken_str,
                    "answers": st.session_state.student_answers,
                    "submitted_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Save complete assessment bundle (graphs, excel, json, pdf report)
                bundle = save_complete_assessment_bundle(
                    pdf_name=data["pdf_name"],
                    all_model_metrics=data["metrics"],
                    all_generated_questions=data["questions"],
                    student_results=student_results,
                    base_output_dir="outputs",
                    config=data["config"],
                    assessment_id=st.session_state.assessment_id,
                    generation_summaries=st.session_state.generation_summaries
                )
                
                # Update student persistent history record
                add_student_assessment_record(curr_user.get("student_id", "STU-2026-8819"), {
                    "assessment_id": bundle.get("assessment_id"),
                    "pdf_name": data["pdf_name"],
                    "score": correct_count,
                    "total": total_q,
                    "percentage": round(pct, 2),
                    "time_taken": time_taken_str,
                    "submitted_at": student_results["submitted_at"],
                    "output_dir": bundle.get("output_dir")
                })
                
                st.session_state.submitted_results = {
                    "student": student_results,
                    "bundle": bundle
                }
                st.session_state.assessment_status = "COMPLETED"
                st.rerun()

    # SUB-VIEW C: Post-Submission Results & Model Comparison
    elif st.session_state.assessment_status == "COMPLETED":
        render_completed_assessment_view()


def render_completed_assessment_view():
    data = st.session_state.generated_data
    sub = st.session_state.submitted_results
    st_res = sub["student"]
    bundle = sub["bundle"]
    metrics = data["metrics"]
    my_model_questions = data["questions"].get("My Model", [])
    
    # Run diagnostic analysis on student answers
    diag_result = diagnose_student_weak_areas(st_res["answers"], my_model_questions)
    topic_breakdown = diag_result["topic_breakdown"]
    weak_topics = diag_result["weak_topics"]
    moderate_topics = diag_result["moderate_topics"]
    strong_topics = diag_result["strong_topics"]
    
    # 1. Student Score & Diagnostics Banner
    st.markdown(f"""
    <div class='edu-header-card'>
        <h2>🎉 Assessment Submitted Successfully</h2>
        <p style='margin-top: 4px; font-size: 1.05rem;'>
            Score: <b>{st_res['score']} / {st_res['total']}</b> ({st_res['percentage']:.1f}%) | 
            Time Taken: <b>{st_res['time_taken']}</b> | 
            Document: <b>{data['pdf_name']}</b> | 
            ID: <code>{bundle.get('assessment_id', '')}</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for structured viewing
    tab_diag, tab_review, tab_compare, tab_graphs, tab_downloads = st.tabs([
        "🎯 Topic Mastery & Weak Areas",
        "📝 Question & Evidence Review",
        "🔬 4-Model Comparative Metrics",
        "📊 Six Evaluation Charts",
        "📥 Download Reports & Data"
    ])
    
    # TAB 1: Topic Mastery & Weak Areas Diagnostic
    with tab_diag:
        st.markdown("<p style='font-size:0.92rem; color:#475569; margin-bottom:16px;'>Diagnostic breakdown of your performance across specific technical topics extracted from the study material. Focus your revision on identified weak areas.</p>", unsafe_allow_html=True)
        
        c_w, c_m, c_s, c_t = st.columns(4)
        with c_w:
            st.markdown(f"""
            <div style='background-color:#FEF2F2; border:1.5px solid #FCA5A5; border-radius:12px; padding:16px; text-align:center;'>
                <div style='font-size:1.8rem; font-weight:800; color:#DC2626;'>{len(weak_topics)}</div>
                <div style='font-size:0.85rem; font-weight:700; color:#991B1B;'>🔴 Weak Areas (<60%)</div>
            </div>
            """, unsafe_allow_html=True)
        with c_m:
            st.markdown(f"""
            <div style='background-color:#FEFCE8; border:1.5px solid #FDE047; border-radius:12px; padding:16px; text-align:center;'>
                <div style='font-size:1.8rem; font-weight:800; color:#CA8A04;'>{len(moderate_topics)}</div>
                <div style='font-size:0.85rem; font-weight:700; color:#854D0E;'>🟡 Moderate (60-79%)</div>
            </div>
            """, unsafe_allow_html=True)
        with c_s:
            st.markdown(f"""
            <div style='background-color:#ECFDF5; border:1.5px solid #6EE7B7; border-radius:12px; padding:16px; text-align:center;'>
                <div style='font-size:1.8rem; font-weight:800; color:#059669;'>{len(strong_topics)}</div>
                <div style='font-size:0.85rem; font-weight:700; color:#065F46;'>🟢 Strong Mastery (≥80%)</div>
            </div>
            """, unsafe_allow_html=True)
        with c_t:
            st.markdown(f"""
            <div style='background-color:#F8FAFC; border:1.5px solid #CBD5E1; border-radius:12px; padding:16px; text-align:center;'>
                <div style='font-size:1.8rem; font-weight:800; color:#475569;'>{len(topic_breakdown)}</div>
                <div style='font-size:0.85rem; font-weight:700; color:#334155;'>📚 Total Topics Evaluated</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        
        # Actionable Weak Topic Alerts
        if weak_topics:
            st.markdown("### ⚠️ Priority Revision Topics (< 60% Accuracy)")
            for wt in weak_topics:
                st.markdown(f"""
                <div style='background-color:#FFF5F5; border:1.5px solid #FEB2B2; border-left:6px solid #E53E3E; border-radius:12px; padding:18px 20px; margin-bottom:14px; box-shadow:0 1px 3px rgba(0,0,0,0.03);'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <h4 style='margin:0; color:#9B2C2C; font-size:1.05rem; font-weight:700;'>📌 {wt['topic']}</h4>
                        <span style='background-color:#FED7D7; color:#C53030; font-weight:800; font-size:0.82rem; padding:4px 10px; border-radius:8px;'>Score: {wt['correct']}/{wt['total']} ({wt['percentage']}%)</span>
                    </div>
                    <div style='margin-top:10px; font-size:0.9rem; color:#2D3748;'>
                        <p style='margin:4px 0;'><b>📖 Relevant Study Pages:</b> <span style='background-color:#E2E8F0; padding:2px 8px; border-radius:6px; font-weight:600;'>Page {wt['pages']}</span></p>
                        <p style='margin:4px 0;'><b>💡 Revision Guidance:</b> {wt['recommendation']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("🌟 Outstanding work! You have achieved strong conceptual mastery across all evaluated topics without any critical weak areas.")
            
        if moderate_topics:
            st.markdown("### 🟡 Topics for Practice & Reinforcement (60-79% Accuracy)")
            for mt in moderate_topics:
                st.markdown(f"""
                <div style='background-color:#FFFFF0; border:1.5px solid #F6E05E; border-left:6px solid #D69E2E; border-radius:12px; padding:16px 18px; margin-bottom:12px;'>
                    <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <h4 style='margin:0; color:#744210; font-size:1.0rem; font-weight:700;'>⚡ {mt['topic']}</h4>
                        <span style='background-color:#FEFCBF; color:#975A16; font-weight:800; font-size:0.82rem; padding:4px 10px; border-radius:8px;'>Score: {mt['correct']}/{mt['total']} ({mt['percentage']}%)</span>
                    </div>
                    <div style='margin-top:8px; font-size:0.88rem; color:#4A5568;'>
                        <p style='margin:2px 0;'><b>📖 Study Pages:</b> Page {mt['pages']} | <b>Advice:</b> {mt['recommendation']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        # Full Topic Breakdown Table
        st.markdown("### 📊 Comprehensive Topic Breakdown")
        df_topic_data = []
        for t in topic_breakdown:
            df_topic_data.append({
                "Topic / Concept": t["topic"],
                "Score": f"{t['correct_answers']} / {t['total_questions']}",
                "Accuracy": f"{t['accuracy_pct']:.1f}%",
                "Status": t["status_label"],
                "Relevant Study Pages": f"Page {t['pages']}",
                "Actionable Advice": t["recommendation"]
            })
        st.dataframe(pd.DataFrame(df_topic_data), use_container_width=True, hide_index=True)

    # TAB 2: Question-by-Question Review with PDF Evidence
    with tab_review:
        st.markdown("<p style='font-size:0.92rem; color:#475569; margin-bottom:16px;'>Detailed post-submission review showing correct answers, your selected choices, explanations, and grounded PDF source excerpts.</p>", unsafe_allow_html=True)
        for idx, q in enumerate(my_model_questions):
            qid = str(q.get("id", idx + 1))
            user_ans = st_res["answers"].get(qid, "Unanswered")
            correct_ans = str(q.get("correct_answer", "")).strip().lower()
            is_correct = str(user_ans).strip().lower() == correct_ans
            
            border_color = "#10B981" if is_correct else "#EF4444"
            badge_bg = "#ECFDF5" if is_correct else "#FEF2F2"
            badge_color = "#059669" if is_correct else "#DC2626"
            status_text = "✓ Correct" if is_correct else "✗ Incorrect"
            
            # Build Options HTML
            options = q.get("options", {})
            opt_html_list = []
            for k in ['a', 'b', 'c', 'd']:
                if k in options:
                    is_correct_opt = (k == correct_ans)
                    is_user_opt = (k == str(user_ans).strip().lower())
                    
                    if is_correct_opt and is_user_opt:
                        card_bg = "#ECFDF5"
                        card_border = "#10B981"
                        badge_html = "<span style='background-color:#059669; color:#FFFFFF; font-size:0.75rem; font-weight:700; padding:4px 10px; border-radius:12px; margin-left:auto;'>✓ Correct (Your Choice)</span>"
                    elif is_correct_opt and not is_user_opt:
                        card_bg = "#F0FDF4"
                        card_border = "#22C55E"
                        badge_html = "<span style='background-color:#16A34A; color:#FFFFFF; font-size:0.75rem; font-weight:700; padding:4px 10px; border-radius:12px; margin-left:auto;'>✓ Correct Answer</span>"
                    elif is_user_opt and not is_correct_opt:
                        card_bg = "#FEF2F2"
                        card_border = "#EF4444"
                        badge_html = "<span style='background-color:#DC2626; color:#FFFFFF; font-size:0.75rem; font-weight:700; padding:4px 10px; border-radius:12px; margin-left:auto;'>✗ Your Choice</span>"
                    else:
                        card_bg = "#FFFFFF"
                        card_border = "#E2E8F0"
                        badge_html = ""
                        
                    opt_html_list.append(
                        f"<div style='background-color:{card_bg}; border:1.5px solid {card_border}; padding:12px 16px; border-radius:10px; margin-bottom:8px; display:flex; align-items:center; box-shadow:0 1px 2px rgba(0,0,0,0.02);'>"
                        f"<span style='font-weight:800; font-size:0.95rem; color:#4F46E5; margin-right:12px; min-width:28px;'>({k.upper()})</span>"
                        f"<span style='font-size:0.92rem; color:#0F172A; font-weight:500; line-height:1.4; flex-grow:1;'>{options[k]}</span>"
                        f"{badge_html}"
                        f"</div>"
                    )
                    
            opt_html = "\n".join(opt_html_list)
            explanation_text = q.get("explanation", "Derived directly from study material.")
            source_page = q.get("source_page", 1)
            source_chunk = q.get("source_chunk", "")[:280]
            
            question_card_html = (
                f"<div style='background-color:#FFFFFF; border:1.5px solid #E2E8F0; border-left:6px solid {border_color}; border-radius:14px; padding:22px 24px; margin-bottom:20px; box-shadow:0 2px 8px rgba(0,0,0,0.04);'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;'>"
                f"<span style='font-weight:700; font-size:1.0rem; color:#1E293B;'>Question {idx + 1}</span>"
                f"<span style='background-color:{badge_bg}; color:{badge_color}; border:1px solid {border_color}; font-weight:700; font-size:0.82rem; padding:4px 12px; border-radius:12px;'>{status_text}</span>"
                f"</div>"
                f"<h3 style='margin:0 0 16px 0; color:#0F172A; font-size:1.15rem; font-weight:700; line-height:1.45;'>{q.get('question')}</h3>"
                f"<div style='margin-bottom:14px;'>"
                f"{opt_html}"
                f"</div>"
                f"<div style='background-color:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:14px 16px; margin-top:12px;'>"
                f"<p style='margin:0 0 4px 0; font-weight:700; font-size:0.85rem; color:#4F46E5;'>💡 Explanation:</p>"
                f"<p style='margin:0 0 10px 0; font-size:0.88rem; color:#334155; line-height:1.4;'>{explanation_text}</p>"
                f"<p style='margin:0 0 4px 0; font-weight:700; font-size:0.82rem; color:#64748B;'>📖 PDF Source Grounding (Page {source_page}):</p>"
                f"<p style='margin:0; font-size:0.84rem; color:#1E293B; font-style:italic; background-color:#EEF2FF; padding:10px 14px; border-radius:8px; border-left:3px solid #6366F1; line-height:1.45;'>\"{source_chunk}...\"</p>"
                f"</div>"
                f"</div>"
            )
            st.markdown(question_card_html, unsafe_allow_html=True)
            
    # TAB 2: Model Comparison Table
    with tab_compare:
        st.markdown("<p style='font-size:0.9rem; color:#64748B;'>Authoritative objective evaluation of all 4 models from the unified metrics table under the 8-check deterministic validation framework.</p>", unsafe_allow_html=True)
        
        # Display authoritative comparative summary table
        df_auth = build_authoritative_metrics_df(metrics)
        df_display = df_auth.copy()
        for col in ["Accuracy", "Precision", "Recall", "F1", "FNR", "FPR"]:
            df_display[col] = df_display[col].apply(lambda v: f"{v:.4f}" if v is not None and not pd.isna(v) else "N/A")
        df_display["TP / FP"] = df_display.apply(lambda r: f"{r['TP']} / {r['FP']}", axis=1)
        st.dataframe(df_display[["Model", "Accuracy", "Precision", "Recall", "F1", "FNR", "FPR", "TP / FP", "Total Questions"]], use_container_width=True, hide_index=True)
        
        # Technical Generation Debug Summary
        with st.expander("📋 Technical Generation & Evaluation Summary", expanded=False):
            gen_sums = st.session_state.get("generation_summaries", {})
            if gen_sums:
                st.markdown("#### Candidate Generation Summary")
                st.dataframe(pd.DataFrame(list(gen_sums.values())), use_container_width=True, hide_index=True)
            st.markdown("#### Mathematical Confusion Matrix Details")
            st.dataframe(df_auth[["Model", "TP", "TN", "FP", "FN", "Total Questions"]], use_container_width=True, hide_index=True)
        
        st.markdown("""
        <div class='edu-card' style='margin-top: 14px; background-color: #F8FAFC;'>
            <h4 style='margin: 0 0 8px 0;'>🔬 Evaluation Methodology</h4>
            <p style='font-size: 0.85rem; color: #475569; margin: 0;'>
                The four models were evaluated on the same uploaded study material using a common question-validity evaluation framework. 
                Questions were assessed for source grounding, answer validity, distractor validity, uniqueness and clarity. 
                Metrics are computed mathematically from True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    # TAB 3: Six Matplotlib Charts (Compact 3x2 Grid)
    with tab_graphs:
        st.markdown("<p style='font-size:0.9rem; color:#64748B;'>Visual comparison of all 4 models across the 6 standardized metrics in a compact 3 × 2 grid.</p>", unsafe_allow_html=True)
        
        g_paths = bundle.get("graphs", {})
        
        # Row 1: Accuracy | Precision
        c1, c2 = st.columns(2)
        with c1:
            if "accuracy" in g_paths and os.path.exists(g_paths["accuracy"]):
                st.image(g_paths["accuracy"], caption="Graph 1: Accuracy Comparison", use_container_width=True)
        with c2:
            if "precision" in g_paths and os.path.exists(g_paths["precision"]):
                st.image(g_paths["precision"], caption="Graph 2: Precision Comparison", use_container_width=True)
                
        # Row 2: Recall | F1 Score
        c3, c4 = st.columns(2)
        with c3:
            if "recall" in g_paths and os.path.exists(g_paths["recall"]):
                st.image(g_paths["recall"], caption="Graph 3: Recall Comparison", use_container_width=True)
        with c4:
            if "f1" in g_paths and os.path.exists(g_paths["f1"]):
                st.image(g_paths["f1"], caption="Graph 4: F1 Score Comparison", use_container_width=True)
                
        # Row 3: FNR | FPR
        c5, c6 = st.columns(2)
        with c5:
            if "fnr" in g_paths and os.path.exists(g_paths["fnr"]):
                st.image(g_paths["fnr"], caption="Graph 5: False Negative Rate (FNR) Comparison", use_container_width=True)
        with c6:
            if "fpr" in g_paths and os.path.exists(g_paths["fpr"]):
                st.image(g_paths["fpr"], caption="Graph 6: False Positive Rate (FPR) Comparison", use_container_width=True)
                
    # TAB 4: Download Reports & Data
    with tab_downloads:
        st.markdown("<p style='font-size:0.9rem; color:#64748B;'>Download formal experiment reports, Excel workbooks, and JSON/CSV evaluation data.</p>", unsafe_allow_html=True)
        
        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            # PDF Report Download
            rep_pdf = bundle.get("report_pdf_path")
            if rep_pdf and os.path.exists(rep_pdf):
                with open(rep_pdf, "rb") as f:
                    st.download_button(
                        label="📄 Download Experiment Report (PDF)",
                        data=f.read(),
                        file_name="experiment_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
            # Excel Download
            xl_path = bundle.get("excel_path")
            if xl_path and os.path.exists(xl_path):
                with open(xl_path, "rb") as f:
                    st.download_button(
                        label="📊 Download Excel Metrics Workbook (.xlsx)",
                        data=f.read(),
                        file_name=os.path.basename(xl_path),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
        with c_dl2:
            # JSON Metrics Download
            j_path = bundle.get("metrics_json")
            if j_path and os.path.exists(j_path):
                with open(j_path, "rb") as f:
                    st.download_button(
                        label="📦 Download Metrics JSON",
                        data=f.read(),
                        file_name="metrics.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
            # CSV Metrics Download
            csv_p = bundle.get("metrics_csv")
            if csv_p and os.path.exists(csv_p):
                with open(csv_p, "rb") as f:
                    st.download_button(
                        label="📑 Download Metrics CSV",
                        data=f.read(),
                        file_name="metrics.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
        st.markdown(f"""
        <div style='margin-top:20px; font-size:0.8rem; color:#64748B;'>
            📁 <b>Saved Local Run Directory:</b> <code>{bundle.get('output_dir')}</code>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height:15px;'></div>", unsafe_allow_html=True)
        if st.button("🚀 Start New Assessment", type="primary", use_container_width=True):
            reset_to_new_assessment()
            st.rerun()


# ============================================================================
# VIEW 3: PERFORMANCE ANALYTICS
# ============================================================================

def render_performance_view():
    st.markdown("""
    <div class='edu-header-card'>
        <h2>📈 Student Performance Analytics</h2>
        <p style='margin-top: 4px;'>Track your study material mastery and question answering accuracy over time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    student = st.session_state.current_user or {}
    s_id = student.get("student_id", "STU-2026-8819")
    history_records = get_student_assessment_history(s_id)
    
    if st.session_state.submitted_results:
        sub = st.session_state.submitted_results["student"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Latest Score", f"{sub['score']} / {sub['total']}")
        with c2:
            st.metric("Accuracy Rate", f"{sub['percentage']:.1f}%")
        with c3:
            st.metric("Time Taken", sub["time_taken"])
            
    if history_records:
        st.markdown("### 📊 Performance History & Score Trends")
        chart_data = []
        for h in reversed(history_records):
            chart_data.append({
                "Date": h.get("submitted_at", "N/A"),
                "Document": h.get("pdf_name", "Test"),
                "Score (%)": h.get("percentage", 0.0)
            })
        df_chart = pd.DataFrame(chart_data)
        st.line_chart(df_chart.set_index("Date")["Score (%)"], color="#4F46E5")
    else:
        if not st.session_state.submitted_results:
            st.info("Complete an assessment to view live performance analytics and progress curves.")


# ============================================================================
# VIEW 4: MODEL COMPARISON LAB
# ============================================================================

def render_model_comparison_view():
    st.markdown("""
    <div class='edu-header-card'>
        <h2>🔬 Model Comparison Laboratory</h2>
        <p style='margin-top: 4px;'>Deep dive into the 4-model evaluation architecture, metrics, and formulation.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='edu-card'>
        <h4>Comparing 4 Pretrained Models & Architectures</h4>
        <table style='width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 10px;'>
            <tr style='background-color: #F1F5F9; border-bottom: 2px solid #CBD5E1;'>
                <th style='padding: 8px;'>Model</th>
                <th style='padding: 8px;'>Architecture Strategy</th>
                <th style='padding: 8px;'>Vector Retrieval</th>
                <th style='padding: 8px;'>Verification Gate</th>
            </tr>
            <tr style='border-bottom: 1px solid #E2E8F0;'>
                <td style='padding: 8px;'><b>My Model</b></td>
                <td style='padding: 8px;'>Multi-stage RAG + Question Blueprinting</td>
                <td style='padding: 8px;'>Dense FAISS (all-MiniLM-L6-v2)</td>
                <td style='padding: 8px;'>8-Check Quality Gate</td>
            </tr>
            <tr style='border-bottom: 1px solid #E2E8F0;'>
                <td style='padding: 8px;'><b>Qwen 2.5 3B</b></td>
                <td style='padding: 8px;'>Qwen/Qwen2.5-3B-Instruct</td>
                <td style='padding: 8px;'>Document Context</td>
                <td style='padding: 8px;'>Standard Generation</td>
            </tr>
            <tr style='border-bottom: 1px solid #E2E8F0;'>
                <td style='padding: 8px;'><b>Phi-3.5 Mini</b></td>
                <td style='padding: 8px;'>microsoft/Phi-3.5-mini-instruct</td>
                <td style='padding: 8px;'>Document Context</td>
                <td style='padding: 8px;'>Standard Generation</td>
            </tr>
            <tr>
                <td style='padding: 8px;'><b>Mistral 7B (4-bit)</b></td>
                <td style='padding: 8px;'>mistralai/Mistral-7B-Instruct-v0.2 (BitsAndBytes NF4)</td>
                <td style='padding: 8px;'>Document Context</td>
                <td style='padding: 8px;'>Standard Generation</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN ROUTING
# ============================================================================

if st.session_state.current_nav == "Dashboard":
    render_dashboard_view()
elif st.session_state.current_nav == "Assessment":
    render_assessment_view()
elif st.session_state.current_nav == "Performance":
    render_performance_view()
elif st.session_state.current_nav == "Model Comparison":
    render_model_comparison_view()
