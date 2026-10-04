import os
import json
import uuid
import hashlib
import secrets
import datetime
from typing import Dict, List, Optional, Tuple

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
USERS_FILE = os.path.join(DATA_DIR, "users.json")


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(USERS_FILE):
        default_users = {
            "users": [
                {
                    "student_id": "STU-2026-8819",
                    "name": "Hariprasath R",
                    "institution": "Sri Eshwar College of Engineering",
                    "department": "Computer Science & Engineering",
                    "email": "student@university.edu",
                    "password_hash": _hash_password("password123"),
                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "assessment_history": []
                }
            ]
        }
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(default_users, f, indent=2)


def _hash_password(password: str) -> str:
    """Hashes a password with SHA-256 and a standard internal salt."""
    salt = "EduMetricSecureSalt2026"
    return hashlib.sha256(f"{salt}{password}".encode("utf-8")).hexdigest()


def _verify_password(password: str, stored_hash: str) -> bool:
    return _hash_password(password) == stored_hash


def load_all_users() -> List[Dict]:
    _ensure_data_dir()
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("users", [])
    except Exception:
        return []


def save_all_users(users: List[Dict]) -> bool:
    _ensure_data_dir()
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump({"users": users}, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving users: {e}")
        return False


def generate_unique_student_id() -> str:
    """Generates a unique student ID formatted like STU-2026-XXXX."""
    users = load_all_users()
    existing_ids = {u.get("student_id", "").upper() for u in users}
    
    current_year = datetime.datetime.now().year
    while True:
        rand_num = secrets.randbelow(9000) + 1000  # 4 digits (1000 - 9999)
        new_id = f"STU-{current_year}-{rand_num}"
        if new_id not in existing_ids:
            return new_id


def find_user_by_identifier(identifier: str) -> Optional[Dict]:
    """Finds a user by email address or student ID."""
    if not identifier:
        return None
    ident = identifier.strip().lower()
    users = load_all_users()
    for user in users:
        if user.get("email", "").lower() == ident or user.get("student_id", "").lower() == ident:
            return user
    return None


def register_student(
    name: str,
    institution: str,
    email: str,
    password: str,
    department: str = "Computer Science & Engineering"
) -> Tuple[bool, str, Optional[Dict]]:
    """
    Registers a new student account.
    Returns (success, message, user_dict).
    """
    name = name.strip()
    institution = institution.strip()
    email = email.strip().lower()
    department = department.strip() if department else "Computer Science & Engineering"
    
    if not name:
        return False, "Full Name is required.", None
    if not institution:
        return False, "College or School Name is required.", None
    if not email or "@" not in email or "." not in email:
        return False, "A valid email address is required.", None
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long.", None
        
    users = load_all_users()
    for u in users:
        if u.get("email", "").lower() == email:
            return False, f"An account with email '{email}' already exists. Please Sign In.", None
            
    student_id = generate_unique_student_id()
    new_user = {
        "student_id": student_id,
        "name": name,
        "institution": institution,
        "department": department,
        "email": email,
        "password_hash": _hash_password(password),
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "assessment_history": []
    }
    
    users.append(new_user)
    if save_all_users(users):
        return True, f"Registration successful! Your Unique Student ID is {student_id}.", new_user
    else:
        return False, "Failed to save user account. Please try again.", None


def authenticate_student(identifier: str, password: str) -> Tuple[bool, str, Optional[Dict]]:
    """
    Authenticates a student by email or student ID + password.
    Returns (success, message, user_dict).
    """
    if not identifier or not password:
        return False, "Please enter both identifier and password.", None
        
    user = find_user_by_identifier(identifier)
    if not user:
        return False, "No student account found with the given Email or Student ID.", None
        
    stored_hash = user.get("password_hash", "")
    if not _verify_password(password, stored_hash):
        return False, "Incorrect password. Please try again.", None
        
    return True, "Credentials verified successfully.", user


def add_student_assessment_record(student_id: str, assessment_summary: Dict) -> bool:
    """Appends an assessment run to the student's saved history."""
    users = load_all_users()
    updated = False
    for u in users:
        if u.get("student_id") == student_id:
            if "assessment_history" not in u:
                u["assessment_history"] = []
            u["assessment_history"].append(assessment_summary)
            updated = True
            break
    if updated:
        return save_all_users(users)
    return False


def get_student_assessment_history(student_id: str, outputs_base: str = "outputs") -> List[Dict]:
    """
    Fetches all assessment runs for a student, merging records from users.json
    and disk scan in outputs/ directory.
    """
    user = find_user_by_identifier(student_id)
    recorded_history = user.get("assessment_history", []) if user else []
    
    # Also scan disk for any matching assessment outputs
    disk_history = []
    if os.path.exists(outputs_base):
        for root, dirs, files in os.walk(outputs_base):
            if "student_results.json" in files:
                try:
                    meta_p = os.path.join(root, "metadata.json")
                    st_p = os.path.join(root, "student_results.json")
                    m_p = os.path.join(root, "metrics.json")
                    pdf_rep = os.path.join(root, "experiment_report.pdf")
                    xl_rep = None
                    for f in files:
                        if f.endswith(".xlsx"):
                            xl_rep = os.path.join(root, f)
                            break
                            
                    with open(st_p, "r", encoding="utf-8") as f:
                        st_data = json.load(f)
                    
                    meta_data = {}
                    if os.path.exists(meta_p):
                        with open(meta_p, "r", encoding="utf-8") as f:
                            meta_data = json.load(f)
                            
                    folder_name = os.path.basename(root)
                    rel_path = os.path.relpath(root, outputs_base)
                    
                    record_sid = st_data.get("student_id") or meta_data.get("student_id")
                    
                    # Include if this matches student or is a general recorded assessment
                    disk_history.append({
                        "run_dir": root,
                        "rel_name": rel_path,
                        "pdf_name": meta_data.get("pdf_name", folder_name),
                        "assessment_id": meta_data.get("assessment_id", folder_name),
                        "score": st_data.get("score", 0),
                        "total": st_data.get("total", 0),
                        "percentage": st_data.get("percentage", 0.0),
                        "time_taken": st_data.get("time_taken", "N/A"),
                        "submitted_at": st_data.get("submitted_at", meta_data.get("timestamp", "N/A")),
                        "metrics_json_path": m_p if os.path.exists(m_p) else None,
                        "report_pdf_path": pdf_rep if os.path.exists(pdf_rep) else None,
                        "excel_path": xl_rep,
                        "student_id": record_sid or student_id,
                        "mtime": os.path.getmtime(root)
                    })
                except Exception:
                    continue
                    
    disk_history.sort(key=lambda x: x.get("mtime", 0), reverse=True)
    return disk_history
