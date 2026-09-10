"""
Resume-JD Alignment Bot (Telegram)
------------------------------------
Flow:
  1. User sends /start
  2. User sends a JD file (PDF/DOCX/TXT)
  3. User sends one or more resume files -> each is scored against the stored JD
  4. /newjd resets and lets the user load a different JD
  5. /compare compares the last two scored resumes
  6. /history lists all resumes scored in the current session

Run:
  pip install -r requirements.txt
  export TELEGRAM_BOT_TOKEN=xxxx
  export NVIDIA_API_KEY=nvapi-xxxx
  python main.py
"""

import os
import io
import re
import json
import logging
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters
)

import pdfplumber
import docx
from openai import OpenAI, RateLimitError

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")

# NVIDIA NIM (build.nvidia.com) — free, OpenAI-compatible endpoint
client: OpenAI | None = None

# Model fallback chain: try primary first, fall back on error
MODELS = [
    "openai/gpt-oss-20b",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
]


def _get_client() -> OpenAI:
    """Lazy-init the OpenAI client so the module can be imported without API keys."""
    global client
    if client is None:
        if not NVIDIA_API_KEY:
            raise RuntimeError("NVIDIA_API_KEY is not set in .env")
        client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
    return client

# In-memory per-chat state:
# { chat_id: {
#     "jd_text": str,
#     "jd_name": str,
#     "resumes": OrderedDict[str, dict]  # filename -> full result JSON
# }}
STATE: dict[int, dict] = {}


# ---------- File parsing ----------

def extract_text(file_bytes: bytes, filename: str) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        chunks = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    chunks.append(t)
        return "\n".join(chunks)
    elif name.endswith(".docx"):
        d = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in d.paragraphs)
    elif name.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file type. Please send a PDF, DOCX, or TXT file.")


# ---------- Score helpers ----------

def _score_band(score: int) -> str:
    """Return a score-band label."""
    if score >= 80:
        return "Strong"
    elif score >= 60:
        return "Moderate"
    return "Weak"


def _score_color(score: int) -> str:
    """Return hex color for a score band."""
    if score >= 80:
        return "#16A34A"  # green
    elif score >= 60:
        return "#D97706"  # amber
    return "#DC2626"      # red


def _score_bar(score: int, width: int = 20) -> str:
    """Monochrome inline bar: ████████░░░░░░░░ 72 Moderate"""
    filled = round(score / 100 * width)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    return f"{bar} {score} {_score_band(score)}"


# ---------- Analysis ----------

ANALYSIS_PROMPT = """You are an ATS (Applicant Tracking System) resume analysis engine.

Given a JOB DESCRIPTION and a RESUME, analyze the alignment.

Return ONLY valid JSON. Do not use markdown. Do not include reasoning.
Do not include ```json. Do not wrap the output in code fences.
Do not infer skills that are not supported by the resume.
Output nothing except the JSON object below.

Schema:

{{
  "ats_score": <integer 0-100>,
  "match_score": <integer 0-100>,
  "matched_skills": [<strings>],
  "missing_skills": [<strings>],
  "strengths": [<strings, 2-4 short bullets>],
  "course_suggestions": [
    {{"skill": "<missing skill>", "course": "<real course/certification name>", "platform": "<e.g. Coursera, Udemy, freeCodeCamp>"}}
  ],
  "verdict": "<one-line hire-worthiness verdict>"
}}

Scoring rules:
- ats_score: weigh keyword/skill overlap with the JD and standard resume structure
  (contact info, clear sections, quantified experience) heavily. This estimates whether
  an automated keyword-based ATS would surface this resume for this JD.
- match_score: weigh actual role/domain/seniority fit, not just keyword overlap.
- If the resume is from a clearly different domain than the JD (e.g. a mechanical
  engineering resume against a software engineering JD), both scores should be low
  and the domain mismatch should be evident in missing_skills.
- List 3-8 items in matched_skills and missing_skills where applicable.
- Provide one course suggestion per missing skill, up to 6 total.
  IMPORTANT: The "skill" field in each course suggestion MUST exactly match one of the
  items you listed in missing_skills. Do not suggest courses for skills not in missing_skills.
- Never invent skills that aren't relevant to the JD.
- Only list skills that are explicitly mentioned or clearly demonstrated in the resume text.

JOB DESCRIPTION:
{jd_text}

RESUME:
{resume_text}
"""


def _extract_json(raw: str) -> dict:
    """Open models sometimes wrap JSON in fences or add a stray sentence;
    strip fences first, then fall back to grabbing the outermost {...}."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            return json.loads(cleaned[start:end + 1])
        raise


def _clamp_scores(result: dict) -> dict:
    """Ensure ats_score and match_score are integers in [0, 100]."""
    for key in ("ats_score", "match_score"):
        val = result.get(key)
        try:
            result[key] = max(0, min(100, int(val)))
        except (TypeError, ValueError):
            result[key] = 0
    return result


def _filter_courses(result: dict) -> dict:
    """Drop any course_suggestions entry whose skill is not a case-insensitive
    substring match of something in missing_skills. This is a code-level guard
    against LLM hallucination of irrelevant courses."""
    missing = [s.lower() for s in result.get("missing_skills", [])]
    kept, dropped = [], []
    for c in result.get("course_suggestions", []):
        skill = (c.get("skill") or "").lower().strip()
        if not skill:
            dropped.append(c)
            continue
        # Accept if the suggestion's skill is a substring of any missing skill or vice-versa
        if any(skill in m or m in skill for m in missing):
            kept.append(c)
        else:
            dropped.append(c)
    for d in dropped:
        logger.debug("Dropped irrelevant course suggestion: %s", d)
    result["course_suggestions"] = kept
    return result


def analyze(jd_text: str, resume_text: str) -> dict:
    """Call the NVIDIA NIM API to score a resume against a JD.
    Tries each model in MODELS in order; falls back on error."""
    prompt_content = ANALYSIS_PROMPT.format(
        jd_text=jd_text[:12000],
        resume_text=resume_text[:12000],
    )
    messages = [{"role": "user", "content": prompt_content}]

    last_error = None
    for model in MODELS:
        try:
            logger.info("Trying model: %s", model)
            resp = _get_client().chat.completions.create(
                model=model,
                temperature=0.2,
                max_tokens=1500,
                messages=messages,
            )
            raw = resp.choices[0].message.content
            result = _extract_json(raw)
            result = _clamp_scores(result)
            result = _filter_courses(result)
            # Ensure list fields default to empty lists
            for field in ("matched_skills", "missing_skills", "strengths", "course_suggestions"):
                if not isinstance(result.get(field), list):
                    result[field] = []
            logger.info("Success with model: %s", model)
            return result
        except RateLimitError:
            raise RuntimeError(
                "Rate limited by NVIDIA API. Please wait a few seconds and resend."
            )
        except json.JSONDecodeError as e:
            logger.warning("Model %s returned unparseable JSON: %s", model, e)
            last_error = e
            continue
        except Exception as e:
            logger.warning("Model %s failed: %s", model, e)
            last_error = e
            continue

    # All models failed
    raise last_error or RuntimeError("All models failed to produce a valid response.")


def format_result(resume_name: str, result: dict) -> str:
    """Format a single resume result as clean, professional Telegram text."""
    ats = result.get("ats_score", 0)
    match = result.get("match_score", 0)

    lines = [
        f"*Resume:* {resume_name}",
        "",
        f"*ATS Score:*  `{_score_bar(ats)}`",
        f"*Match Score:* `{_score_bar(match)}`",
        "",
    ]

    matched = result.get("matched_skills", [])
    lines.append("*Matched Skills:*")
    lines.append(", ".join(matched) if matched else "None found")
    lines.append("")

    missing = result.get("missing_skills", [])
    lines.append("*Skill Gaps:*")
    lines.append(", ".join(missing) if missing else "None")
    lines.append("")

    if result.get("strengths"):
        lines.append("*Strengths:*")
        for s in result["strengths"]:
            lines.append(f"  - {s}")
        lines.append("")

    if result.get("course_suggestions"):
        lines.append("*Recommended Courses:*")
        for c in result["course_suggestions"]:
            lines.append(f"  - {c.get('skill')} -> {c.get('course')} ({c.get('platform')})")
        lines.append("")

    lines.append(f"*Verdict:* {result.get('verdict', 'N/A')}")
    return "\n".join(lines)


# ---------- Chart generation ----------

def _generate_chart(resumes: dict) -> bytes:
    """Generate a professional horizontal bar chart comparing resume scores.
    Returns PNG bytes."""
    names = list(resumes.keys())
    ats_scores = [resumes[n].get("ats_score", 0) for n in names]
    match_scores = [resumes[n].get("match_score", 0) for n in names]

    # Truncate long filenames for readability
    labels = [n[:25] + "..." if len(n) > 28 else n for n in names]

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6, max(3, len(names) * 1.2)), dpi=150)
    fig.patch.set_facecolor("#F8FAFC")
    ax.set_facecolor("#F8FAFC")

    bar_height = 0.35
    y_positions = range(len(names))

    # ATS bars
    ats_colors = [_score_color(s) for s in ats_scores]
    bars_ats = ax.barh(
        [y - bar_height / 2 for y in y_positions],
        ats_scores, bar_height, label="ATS Score", color=ats_colors, edgecolor="white"
    )
    ax.bar_label(bars_ats, fmt="%d", padding=3, fontsize=9, color="#0F172A")

    # Match bars
    match_colors = [_score_color(s) for s in match_scores]
    bars_match = ax.barh(
        [y + bar_height / 2 for y in y_positions],
        match_scores, bar_height, label="Match Score", color=match_colors,
        edgecolor="white", hatch="///"
    )
    ax.bar_label(bars_match, fmt="%d", padding=3, fontsize=9, color="#0F172A")

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, fontsize=10, color="#0F172A")
    ax.set_xlabel("Score (0–100)", fontsize=10, color="#0F172A")
    ax.set_xlim(0, 110)
    ax.set_title("Resume Scores", fontsize=12, fontweight="bold", color="#0F172A", pad=10)

    # Distinguish ATS (solid) vs Match (hatched) with a minimal legend
    ax.legend(fontsize=8, loc="lower right", framealpha=0.7)

    # Light gridlines
    ax.xaxis.grid(True, linestyle=":", linewidth=0.5, color="#CBD5E1")
    ax.set_axisbelow(True)
    ax.tick_params(colors="#0F172A")

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#F8FAFC")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------- Comparison logic (no LLM) ----------

def _build_comparison(resumes: dict, name_a: str, name_b: str) -> str:
    """Build a professional comparison block from two cached resume results.
    No LLM call — pure Python over structured data."""
    a = resumes[name_a]
    b = resumes[name_b]

    a_ats, a_match = a.get("ats_score", 0), a.get("match_score", 0)
    b_ats, b_match = b.get("ats_score", 0), b.get("match_score", 0)

    # Determine winner by match_score, tie-break on ats_score
    if a_match > b_match or (a_match == b_match and a_ats > b_ats):
        winner, loser = name_a, name_b
        w, l = a, b
    elif b_match > a_match or (b_match == a_match and b_ats > a_ats):
        winner, loser = name_b, name_a
        w, l = b, a
    else:
        winner, loser = name_a, name_b  # exact tie — arbitrary pick
        w, l = a, b

    # Truncate names for the table
    short_a = name_a[:20] + ".." if len(name_a) > 22 else name_a
    short_b = name_b[:20] + ".." if len(name_b) > 22 else name_b

    # Build table
    header = f"{'Metric':<16} {short_a:>22} {short_b:>22}"
    sep = "-" * len(header)
    row_ats = f"{'ATS Score':<16} {a_ats:>22} {b_ats:>22}"
    row_match = f"{'Match Score':<16} {a_match:>22} {b_match:>22}"
    row_band = f"{'Band':<16} {_score_band(a_match):>22} {_score_band(b_match):>22}"

    table = f"```\n{header}\n{sep}\n{row_ats}\n{row_match}\n{row_band}\n```"

    # Strengths / weaknesses from cached data
    lines = [
        "*Resume Comparison*",
        "",
        table,
        "",
        f"*Stronger fit:* {winner}",
    ]

    w_match = w.get("match_score", 0)
    l_match = l.get("match_score", 0)
    diff = w_match - l_match
    if diff > 0:
        lines.append(f"  {winner} scores {diff} points higher on match score.")
    else:
        lines.append("  Scores are tied; edge given on ATS score.")
    lines.append("")

    # Per-candidate detail
    for label, name, res in [(name_a, short_a, a), (name_b, short_b, b)]:
        strengths = res.get("strengths", [])
        matched = res.get("matched_skills", [])
        missing = res.get("missing_skills", [])

        lines.append(f"*{name}*")
        good = matched[:5]
        lines.append(f"  Strong at: {', '.join(good) if good else 'N/A'}")
        if strengths:
            for s in strengths[:3]:
                lines.append(f"    - {s}")
        weak = missing[:5]
        lines.append(f"  Gaps: {', '.join(weak) if weak else 'None'}")
        lines.append("")

    return "\n".join(lines)


# ---------- Telegram handlers ----------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    STATE.pop(chat_id, None)
    await update.message.reply_text(
        "*Resume-JD Alignment Bot*\n\n"
        "Step 1: Send the *Job Description* file (PDF/DOCX/TXT).\n"
        "Step 2: Send one or more *Resume* files to score against the JD.\n\n"
        "Commands:\n"
        "  /newjd  — Reset and load a different JD\n"
        "  /compare — Compare last two scored resumes\n"
        "  /history — List all resumes scored this session",
        parse_mode="Markdown",
    )


async def newjd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    STATE.pop(chat_id, None)
    await update.message.reply_text("Reset complete. Send the new Job Description file.")


async def compare_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    state = STATE.get(chat_id)
    resumes = state.get("resumes", OrderedDict()) if state else OrderedDict()

    if len(resumes) < 2:
        await update.message.reply_text(
            "Need at least 2 scored resumes to compare. "
            "Upload more resume files first."
        )
        return

    # Pick last two
    keys = list(resumes.keys())
    name_a, name_b = keys[-2], keys[-1]

    try:
        comparison_text = _build_comparison(resumes, name_a, name_b)
        await update.message.reply_text(comparison_text, parse_mode="Markdown")

        # Send comparison chart
        chart_resumes = OrderedDict()
        chart_resumes[name_a] = resumes[name_a]
        chart_resumes[name_b] = resumes[name_b]
        chart_bytes = _generate_chart(chart_resumes)
        await update.message.reply_photo(
            photo=chart_bytes, caption="Score comparison"
        )
    except Exception as e:
        logger.exception("Comparison failed")
        await update.message.reply_text(f"Comparison failed: {e}")


async def history_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    state = STATE.get(chat_id)
    resumes = state.get("resumes", OrderedDict()) if state else OrderedDict()

    if not resumes:
        await update.message.reply_text("No resumes scored in this session yet.")
        return

    jd_name = state.get("jd_name", "Unknown JD")
    lines = [f"*Session History* (JD: {jd_name})", ""]

    header = f"{'#':<4} {'Resume':<28} {'ATS':>5} {'Match':>5} {'Band':<10}"
    lines.append(f"```\n{header}")
    lines.append("-" * len(header))

    for i, (name, res) in enumerate(resumes.items(), 1):
        short = name[:26] + ".." if len(name) > 28 else name
        ats = res.get("ats_score", 0)
        match = res.get("match_score", 0)
        band = _score_band(match)
        lines.append(f"{i:<4} {short:<28} {ats:>5} {match:>5} {band:<10}")

    lines.append("```")
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    doc = update.message.document
    filename = doc.file_name or "file"

    try:
        tg_file = await context.bot.get_file(doc.file_id)
        file_bytes = await tg_file.download_as_bytearray()
        text = extract_text(bytes(file_bytes), filename)
    except Exception as e:
        await update.message.reply_text(f"Could not read {filename}: {e}")
        return

    if not text.strip():
        await update.message.reply_text(f"No extractable text found in {filename}.")
        return

    state = STATE.get(chat_id)

    # First file → treat as JD
    if state is None or "jd_text" not in state:
        STATE[chat_id] = {"jd_text": text, "jd_name": filename, "resumes": OrderedDict()}
        await update.message.reply_text(
            f"JD received: *{filename}*\nNow send one or more resumes to score against it.",
            parse_mode="Markdown",
        )
        return

    # Subsequent files → treat as resumes
    await update.message.reply_text(f"Analyzing *{filename}* against *{state['jd_name']}*...",
                                    parse_mode="Markdown")
    try:
        result = analyze(state["jd_text"], text)
    except RuntimeError as e:
        # Catches our wrapped RateLimitError
        await update.message.reply_text(str(e))
        return
    except json.JSONDecodeError:
        await update.message.reply_text(
            "Could not parse the AI response. Please resend the resume."
        )
        return
    except Exception as e:
        logger.exception("Analysis failed")
        await update.message.reply_text(f"Analysis failed for {filename}: {e}")
        return

    # Cache result
    state.setdefault("resumes", OrderedDict())[filename] = result

    # Send text result
    msg = format_result(filename, result)
    await update.message.reply_text(msg, parse_mode="Markdown")

    # Send chart for this resume
    try:
        chart_data = OrderedDict()
        chart_data[filename] = result
        chart_bytes = _generate_chart(chart_data)
        await update.message.reply_photo(photo=chart_bytes, caption="Score breakdown")
    except Exception as e:
        logger.warning("Chart generation failed: %s", e)

    # Hint about /compare if 2+ resumes
    resume_count = len(state.get("resumes", {}))
    if resume_count == 2:
        await update.message.reply_text(
            f"{resume_count} resumes scored. Use /compare to compare them."
        )
    elif resume_count > 2:
        await update.message.reply_text(
            f"{resume_count} resumes scored. Use /compare to compare the last two, "
            "or /history to list all."
        )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Please send a PDF, DOCX, or TXT file (JD first, then resumes). Use /start for instructions."
    )


def main():
    if not TELEGRAM_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set in .env")
    if not NVIDIA_API_KEY:
        raise SystemExit("NVIDIA_API_KEY is not set in .env")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("newjd", newjd))
    app.add_handler(CommandHandler("compare", compare_cmd))
    app.add_handler(CommandHandler("history", history_cmd))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("Bot starting (polling)...")
    app.run_polling()


if __name__ == "__main__":
    main()
