import os
import json
import smtplib
import secrets
import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, Optional, Tuple

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config"))
SMTP_CONFIG_FILE = os.path.join(CONFIG_DIR, "smtp_config.json")


def _ensure_config_dir():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.exists(SMTP_CONFIG_FILE):
        default_cfg = {
            "smtp_host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
            "smtp_port": int(os.getenv("SMTP_PORT", 587)),
            "smtp_user": os.getenv("SMTP_USER", ""),
            "smtp_password": os.getenv("SMTP_PASSWORD", ""),
            "smtp_use_tls": True,
            "sender_name": "EduMetric AI Security"
        }
        with open(SMTP_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(default_cfg, f, indent=2)


def get_smtp_config() -> Dict:
    """Loads SMTP configuration from config file or environment variables."""
    _ensure_config_dir()
    try:
        with open(SMTP_CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        cfg = {}
        
    # Override with env vars if set
    if os.getenv("SMTP_HOST"):
        cfg["smtp_host"] = os.getenv("SMTP_HOST")
    if os.getenv("SMTP_PORT"):
        try:
            cfg["smtp_port"] = int(os.getenv("SMTP_PORT"))
        except ValueError:
            pass
    if os.getenv("SMTP_USER"):
        cfg["smtp_user"] = os.getenv("SMTP_USER")
    if os.getenv("SMTP_PASSWORD"):
        cfg["smtp_password"] = os.getenv("SMTP_PASSWORD")
        
    return cfg


def save_smtp_config(host: str, port: int, user: str, password: str, use_tls: bool = True) -> bool:
    """Saves SMTP credentials to smtp_config.json."""
    _ensure_config_dir()
    cfg = {
        "smtp_host": host.strip() if host else "smtp.gmail.com",
        "smtp_port": int(port) if port else 587,
        "smtp_user": user.strip(),
        "smtp_password": password.strip(),
        "smtp_use_tls": bool(use_tls),
        "sender_name": "EduMetric AI Security"
    }
    try:
        with open(SMTP_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving SMTP config: {e}")
        return False


def generate_secure_otp() -> str:
    """Generates a cryptographically strong 6-digit numeric OTP."""
    return f"{secrets.randbelow(900000) + 100000}"


def mask_email(email: str) -> str:
    """Masks an email address for privacy (e.g. j***e@domain.com)."""
    if not email or "@" not in email:
        return email
    local, domain = email.split("@", 1)
    if len(local) <= 2:
        masked_local = local[0] + "*"
    else:
        masked_local = local[0] + "*" * (len(local) - 2) + local[-1]
    return f"{masked_local}@{domain}"


def send_otp_via_email(
    recipient_email: str,
    recipient_name: str,
    otp_code: str,
    student_id: Optional[str] = None
) -> Dict:
    """
    Sends the 6-digit OTP code to the student's email address via SMTP.
    Returns status dictionary with 'success', 'message', 'masked_email', 'sent_real_email', and 'error'.
    """
    cfg = get_smtp_config()
    smtp_host = cfg.get("smtp_host", "smtp.gmail.com")
    smtp_port = int(cfg.get("smtp_port", 587))
    smtp_user = cfg.get("smtp_user", "").strip()
    smtp_password = cfg.get("smtp_password", "").strip()
    smtp_use_tls = cfg.get("smtp_use_tls", True)
    sender_name = cfg.get("sender_name", "EduMetric AI Security")
    
    masked = mask_email(recipient_email)
    
    # HTML Email Template
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #F8FAFC; margin: 0; padding: 20px; }}
            .container {{ max-width: 540px; margin: 0 auto; background: #FFFFFF; border-radius: 12px; border: 1px solid #E2E8F0; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }}
            .header {{ background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%); padding: 24px; text-align: center; color: #FFFFFF; }}
            .header h1 {{ margin: 0; font-size: 22px; font-weight: 700; }}
            .content {{ padding: 28px 24px; color: #1E293B; }}
            .otp-box {{ background-color: #EEF2FF; border: 2px dashed #6366F1; border-radius: 10px; padding: 18px; text-align: center; margin: 24px 0; }}
            .otp-code {{ font-size: 32px; font-weight: 800; letter-spacing: 6px; color: #4338CA; font-family: 'Courier New', monospace; }}
            .badge {{ display: inline-block; background-color: #ECFDF5; color: #059669; font-size: 12px; font-weight: 700; padding: 4px 10px; border-radius: 12px; margin-top: 6px; }}
            .footer {{ background-color: #F8FAFC; border-top: 1px solid #E2E8F0; padding: 16px 24px; text-align: center; font-size: 12px; color: #64748B; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎓 EduMetric AI Verification</h1>
                <p style="margin: 6px 0 0 0; font-size: 13px; opacity: 0.9;">Secure 2-Step Authentication Gateway</p>
            </div>
            <div class="content">
                <p style="font-size: 15px; margin-top: 0;">Hello <b>{recipient_name}</b>,</p>
                <p style="font-size: 14px; color: #475569; line-height: 1.5;">
                    You recently requested access to the <b>EduMetric AI Assessment Portal</b> for Student ID: <code>{student_id or 'N/A'}</code>.
                    Please use the following 6-digit verification code to complete your login:
                </p>
                
                <div class="otp-box">
                    <div style="font-size: 12px; color: #6366F1; font-weight: 600; text-transform: uppercase; margin-bottom: 4px;">One-Time Verification Code</div>
                    <div class="otp-code">{otp_code}</div>
                    <div class="badge">⏱️ Valid for 10 minutes</div>
                </div>
                
                <p style="font-size: 13px; color: #64748B; line-height: 1.4; margin-bottom: 0;">
                    ⚠️ <b>Security Notice:</b> Never share this OTP with anyone. EduMetric staff will never ask for your code. If you did not request this login, please ignore this email.
                </p>
            </div>
            <div class="footer">
                EduMetric AI Assessment Platform • Automated PDF-Grounded Evaluation Engine
            </div>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
    EduMetric AI - 2-Step Verification Code
    --------------------------------------
    Hello {recipient_name},

    Your 2-Step Verification Code for EduMetric AI is:
    OTP: {otp_code}

    Student ID: {student_id or 'N/A'}
    Valid for: 10 minutes

    If you did not request this verification code, please ignore this email.
    """
    
    # If SMTP credentials are configured, send real email
    if smtp_user and smtp_password:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"🔑 EduMetric AI Verification Code: {otp_code}"
            msg["From"] = f"{sender_name} <{smtp_user}>"
            msg["To"] = recipient_email
            
            msg.attach(MIMEText(text_body, "plain"))
            msg.attach(MIMEText(html_body, "html"))
            
            if smtp_port == 465:
                server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=10)
            else:
                server = smtplib.SMTP(smtp_host, smtp_port, timeout=10)
                if smtp_use_tls:
                    server.starttls()
                    
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [recipient_email], msg.as_string())
            server.quit()
            
            return {
                "success": True,
                "message": f"Verification code successfully sent to {recipient_email}.",
                "masked_email": masked,
                "sent_real_email": True,
                "error": None
            }
        except smtplib.SMTPAuthenticationError as e:
            return {
                "success": False,
                "message": "Gmail/SMTP Authentication failed. If using Gmail, please create a 16-character App Password at myaccount.google.com/apppasswords.",
                "masked_email": masked,
                "sent_real_email": False,
                "error": "AUTH_ERROR"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Could not connect to SMTP mail server: {str(e)}",
                "masked_email": masked,
                "sent_real_email": False,
                "error": str(e)
            }
    else:
        # SMTP not configured yet
        return {
            "success": False,
            "message": "SMTP sender account is not yet configured. Please configure your Gmail App Password below to send live emails.",
            "masked_email": masked,
            "sent_real_email": False,
            "error": "SMTP_NOT_CONFIGURED"
        }
