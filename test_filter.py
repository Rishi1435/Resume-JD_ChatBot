"""
Unit tests for Resume-JD Alignment Bot helpers.
Run:  pytest test_filter.py -v
"""

import json
import pytest
from collections import OrderedDict

# Import helpers from main module
from main import (
    _filter_courses, _clamp_scores, _score_band, _build_comparison,
    classify_document,
)


# ---------- Course filter tests ----------

class TestFilterCourses:
    """Verify post-LLM course filter drops irrelevant suggestions."""

    def test_irrelevant_course_dropped(self):
        """Java course should be dropped when missing_skills only has Python."""
        result = {
            "missing_skills": ["Python", "Data Analysis"],
            "course_suggestions": [
                {"skill": "Python", "course": "Python for Everybody", "platform": "Coursera"},
                {"skill": "Java", "course": "Java Masterclass", "platform": "Udemy"},
            ],
        }
        filtered = _filter_courses(result)
        assert len(filtered["course_suggestions"]) == 1
        assert filtered["course_suggestions"][0]["skill"] == "Python"

    def test_case_insensitive_match(self):
        """Filter should be case-insensitive."""
        result = {
            "missing_skills": ["machine learning"],
            "course_suggestions": [
                {"skill": "Machine Learning", "course": "ML Course", "platform": "Coursera"},
            ],
        }
        filtered = _filter_courses(result)
        assert len(filtered["course_suggestions"]) == 1

    def test_substring_match(self):
        """'SQL' should match 'SQL Database Management'."""
        result = {
            "missing_skills": ["SQL Database Management"],
            "course_suggestions": [
                {"skill": "SQL", "course": "SQL Bootcamp", "platform": "Udemy"},
            ],
        }
        filtered = _filter_courses(result)
        assert len(filtered["course_suggestions"]) == 1

    def test_empty_missing_skills_drops_all(self):
        """If no missing skills, all courses should be dropped."""
        result = {
            "missing_skills": [],
            "course_suggestions": [
                {"skill": "Python", "course": "Python 101", "platform": "edX"},
            ],
        }
        filtered = _filter_courses(result)
        assert len(filtered["course_suggestions"]) == 0

    def test_empty_courses_stays_empty(self):
        """No courses in → no courses out, no crash."""
        result = {
            "missing_skills": ["Python"],
            "course_suggestions": [],
        }
        filtered = _filter_courses(result)
        assert filtered["course_suggestions"] == []

    def test_missing_skill_field_in_course(self):
        """Course with None/missing skill field should be dropped."""
        result = {
            "missing_skills": ["Python"],
            "course_suggestions": [
                {"skill": None, "course": "Random Course", "platform": "Udemy"},
                {"course": "Another Course", "platform": "Udemy"},
            ],
        }
        filtered = _filter_courses(result)
        assert len(filtered["course_suggestions"]) == 0


# ---------- Score clamping tests ----------

class TestClampScores:
    def test_normal_scores_unchanged(self):
        result = {"ats_score": 75, "match_score": 60}
        clamped = _clamp_scores(result)
        assert clamped["ats_score"] == 75
        assert clamped["match_score"] == 60

    def test_over_100_clamped(self):
        result = {"ats_score": 150, "match_score": 200}
        clamped = _clamp_scores(result)
        assert clamped["ats_score"] == 100
        assert clamped["match_score"] == 100

    def test_negative_clamped(self):
        result = {"ats_score": -10, "match_score": -5}
        clamped = _clamp_scores(result)
        assert clamped["ats_score"] == 0
        assert clamped["match_score"] == 0

    def test_non_numeric_defaults_to_zero(self):
        result = {"ats_score": "high", "match_score": None}
        clamped = _clamp_scores(result)
        assert clamped["ats_score"] == 0
        assert clamped["match_score"] == 0

    def test_float_scores_truncated(self):
        result = {"ats_score": 72.8, "match_score": 55.2}
        clamped = _clamp_scores(result)
        assert clamped["ats_score"] == 72
        assert clamped["match_score"] == 55


# ---------- Score band tests ----------

class TestScoreBand:
    def test_strong(self):
        assert _score_band(80) == "Strong"
        assert _score_band(100) == "Strong"

    def test_moderate(self):
        assert _score_band(60) == "Moderate"
        assert _score_band(79) == "Moderate"

    def test_weak(self):
        assert _score_band(59) == "Weak"
        assert _score_band(0) == "Weak"


# ---------- Comparison tests ----------

class TestBuildComparison:
    def _make_resumes(self):
        """Create two cached resume results: one strong, one weak."""
        resumes = OrderedDict()
        resumes["alice_resume.pdf"] = {
            "ats_score": 85,
            "match_score": 90,
            "matched_skills": ["Python", "Machine Learning", "SQL", "TensorFlow"],
            "missing_skills": ["Kubernetes"],
            "strengths": ["5 years ML experience", "Published research papers"],
            "course_suggestions": [
                {"skill": "Kubernetes", "course": "K8s for Devs", "platform": "Udemy"}
            ],
            "verdict": "Strong fit for the role",
        }
        resumes["bob_resume.pdf"] = {
            "ats_score": 40,
            "match_score": 35,
            "matched_skills": ["Python"],
            "missing_skills": ["Machine Learning", "SQL", "TensorFlow", "Kubernetes"],
            "strengths": ["Basic Python knowledge"],
            "course_suggestions": [
                {"skill": "Machine Learning", "course": "ML Crash Course", "platform": "Google"}
            ],
            "verdict": "Weak fit — significant skill gaps",
        }
        return resumes

    def test_stronger_candidate_wins(self):
        resumes = self._make_resumes()
        text = _build_comparison(resumes, "alice_resume.pdf", "bob_resume.pdf")
        assert "alice_resume.pdf" in text
        # Alice should be identified as the stronger fit
        # The "Stronger fit:" line should reference Alice
        assert "alice_resume" in text.split("Stronger fit:")[1].split("\n")[0]

    def test_comparison_contains_scores(self):
        resumes = self._make_resumes()
        text = _build_comparison(resumes, "alice_resume.pdf", "bob_resume.pdf")
        assert "85" in text  # Alice ATS
        assert "90" in text  # Alice match
        assert "40" in text  # Bob ATS
        assert "35" in text  # Bob match

    def test_comparison_grounded_in_skills(self):
        resumes = self._make_resumes()
        text = _build_comparison(resumes, "alice_resume.pdf", "bob_resume.pdf")
        # Alice's matched skills should appear
        assert "Python" in text
        assert "Machine Learning" in text
        # Bob's gaps should appear
        assert "TensorFlow" in text

    def test_tied_scores_no_crash(self):
        resumes = OrderedDict()
        resumes["a.pdf"] = {
            "ats_score": 70, "match_score": 70,
            "matched_skills": ["A"], "missing_skills": ["B"],
            "strengths": ["X"], "course_suggestions": [], "verdict": "OK",
        }
        resumes["b.pdf"] = {
            "ats_score": 70, "match_score": 70,
            "matched_skills": ["C"], "missing_skills": ["D"],
            "strengths": ["Y"], "course_suggestions": [], "verdict": "OK",
        }
        text = _build_comparison(resumes, "a.pdf", "b.pdf")
        assert "Stronger fit:" in text


# ---------- Document classification tests ----------

class TestClassifyDocument:
    """Verify document auto-detection properly identifies JDs vs Resumes."""

    def test_classify_user_resume_filename(self):
        """Rishi_Resume_Software_developer_.pdf should be classified as resume."""
        doc_type = classify_document("Rishi_Resume_Software_developer_.pdf", "Python developer")
        assert doc_type == "resume"

    def test_classify_user_jd_filename(self):
        """DLMLE Intern JD 2027.pdf should be classified as jd."""
        doc_type = classify_document("DLMLE Intern JD 2027.pdf", "Machine Learning internship requirements")
        assert doc_type == "jd"

    def test_classify_cv_filename(self):
        """John_Doe_CV.docx should be classified as resume."""
        doc_type = classify_document("John_Doe_CV.docx", "")
        assert doc_type == "resume"

    def test_classify_job_description_filename(self):
        """Senior_Backend_Job_Description.pdf should be classified as jd."""
        doc_type = classify_document("Senior_Backend_Job_Description.pdf", "")
        assert doc_type == "jd"

    def test_classify_by_content_when_filename_generic(self):
        """Generic filename 'document.pdf' classified by resume content cues."""
        resume_text = """
        John Doe
        john.doe@example.com | +1 555-0199 | github.com/johndoe | linkedin.com/in/johndoe
        Education:
        B.Tech in Computer Science, GPA 3.8, ABC University
        Professional Experience:
        Software Engineer at Acme Corp (2022-Present)
        Technical Skills:
        Python, C++, AWS, Docker, Kubernetes
        Personal Projects:
        Developed automated ATS scanner
        """
        assert classify_document("document.pdf", resume_text) == "resume"

    def test_classify_by_content_when_jd_content(self):
        """Generic filename 'doc1.pdf' classified by JD content cues."""
        jd_text = """
        About the Role:
        We are seeking a Machine Learning Engineer to join our team.
        Key Responsibilities:
        - Design and deploy machine learning models in production.
        - Collaborate with backend engineers to integrate APIs.
        Minimum Qualifications:
        - 3+ years of experience in Python and PyTorch.
        What We Offer:
        - Competitive salary, health benefits, and 401(k) matching.
        Apply now by submitting your resume.
        """
        assert classify_document("doc1.pdf", jd_text) == "jd"

    def test_caption_override_jd(self):
        """Explicit caption 'jd' forces document classification to jd."""
        assert classify_document("ambiguous.pdf", "Some random text", caption="jd") == "jd"

    def test_caption_override_resume(self):
        """Explicit caption 'resume' forces document classification to resume."""
        assert classify_document("ambiguous.pdf", "Some random text", caption="resume") == "resume"
