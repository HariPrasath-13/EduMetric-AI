import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.auth_service import (
    register_student,
    authenticate_student,
    generate_unique_student_id,
    find_user_by_identifier,
    add_student_assessment_record,
    get_student_assessment_history,
    load_all_users
)
from utils.email_service import (
    generate_secure_otp,
    mask_email,
    send_otp_via_email,
    get_smtp_config,
    save_smtp_config
)


def run_auth_and_dashboard_tests():
    print("==================================================")
    print("RUNNING AUTHENTICATION & DASHBOARD TESTS")
    print("==================================================")
    
    # 1. Test unique student ID generation
    print("\n[TEST 1] Testing unique student ID format and uniqueness...")
    id1 = generate_unique_student_id()
    id2 = generate_unique_student_id()
    assert id1.startswith("STU-2026-"), f"Expected ID starting with STU-2026-, got {id1}"
    assert id2.startswith("STU-2026-"), f"Expected ID starting with STU-2026-, got {id2}"
    assert id1 != id2, "Generated IDs should be distinct"
    print(f"[PASS] Unique IDs generated: {id1}, {id2}")
    
    # 2. Test registration
    print("\n[TEST 2] Testing student registration flow...")
    test_email = f"test_student_{int(time.time())}@university.edu"
    ok, msg, user = register_student(
        name="Alexander Wright",
        institution="Stanford University",
        email=test_email,
        password="securepassword123",
        department="Computer Science"
    )
    assert ok is True, f"Registration failed: {msg}"
    assert user is not None
    assert user["student_id"].startswith("STU-2026-")
    assert user["name"] == "Alexander Wright"
    assert user["institution"] == "Stanford University"
    assert user["email"] == test_email
    print(f"[PASS] Registered student: {user['name']} with ID: {user['student_id']}")
    
    # 3. Test duplicate email prevention
    print("\n[TEST 3] Testing duplicate email registration prevention...")
    dup_ok, dup_msg, _ = register_student(
        name="Alex Duplicate",
        institution="Stanford University",
        email=test_email,
        password="password123"
    )
    assert dup_ok is False, "Duplicate email registration should fail"
    print(f"[PASS] Duplicate properly rejected with message: {dup_msg}")
    
    # 4. Test authentication by Email and Student ID
    print("\n[TEST 4] Testing authentication by Email and by Student ID...")
    auth_ok_email, _, user_by_email = authenticate_student(test_email, "securepassword123")
    assert auth_ok_email is True
    assert user_by_email["student_id"] == user["student_id"]
    
    auth_ok_id, _, user_by_id = authenticate_student(user["student_id"], "securepassword123")
    assert auth_ok_id is True
    assert user_by_id["email"] == test_email
    
    # Invalid password check
    bad_auth, _, _ = authenticate_student(test_email, "wrongpassword")
    assert bad_auth is False
    print("[PASS] Authentication successfully verified for Email and Student ID.")
    
    # 5. Test OTP Generation and Email Masking
    print("\n[TEST 5] Testing OTP generation and email masking...")
    otp = generate_secure_otp()
    assert len(otp) == 6 and otp.isdigit(), f"Invalid OTP: {otp}"
    masked = mask_email("alexander.wright@stanford.edu")
    assert "@" in masked and masked.startswith("a*")
    print(f"[PASS] Generated 6-digit OTP: {otp} | Masked email: {masked}")
    
    # 6. Test OTP Dispatch Simulation / Safe handling
    print("\n[TEST 6] Testing email dispatch handling...")
    email_res = send_otp_via_email(
        recipient_email=test_email,
        recipient_name=user["name"],
        otp_code=otp,
        student_id=user["student_id"]
    )
    assert "masked_email" in email_res
    print(f"[PASS] Email dispatch handled safely: {email_res['message']}")
    
    # 7. Test Student Assessment Record & History Retrieval
    print("\n[TEST 7] Testing student assessment history recording and retrieval...")
    add_student_assessment_record(user["student_id"], {
        "assessment_id": "assessment_20261004_test1",
        "pdf_name": "Distributed_Systems_Unit_1.pdf",
        "score": 5,
        "total": 5,
        "percentage": 100.0,
        "time_taken": "2m 10s",
        "submitted_at": "2026-10-04 11:30:00"
    })
    
    history = get_student_assessment_history(user["student_id"])
    assert len(history) >= 1
    print(f"[PASS] Student history retrieved with {len(history)} records.")
    
    print("\n==================================================")
    print("ALL AUTHENTICATION & DASHBOARD TESTS PASSED!")
    print("==================================================")


if __name__ == "__main__":
    run_auth_and_dashboard_tests()
