# Resume-JD Alignment Bot

A high-performance, professional Telegram bot that analyzes resumes against Job Descriptions (JDs) using NVIDIA NIM inference endpoints and OpenAI-compatible reasoning models.

## Key Features

- **Document Parsing**: Extracts text from PDF (`pdfplumber`), DOCX (`python-docx`), and plain text (`TXT`) files.
- **AI Scoring & Gap Analysis**:
  - ATS Compatibility Score (0–100) and JD Match Score (0–100)
  - Matched core competencies and missing skill gaps
  - Course recommendations filtered strictly to address identified skill gaps
  - Executive hiring verdict (`Strong Match`, `Moderate Match`, `Weak Match`)
- **Visual Analytics**: Dynamic, professional Matplotlib score-band bar charts sent directly via Telegram.
- **Multi-Resume Comparison (`/compare`)**: Side-by-side formatted monospace alignment table comparing all resumes evaluated against the active JD.
- **Session History (`/history`)**: Review past evaluations with timestamps and ATS scores.
- **Resilient Model Fallback Chain**:
  1. `openai/gpt-oss-20b` (Primary)
  2. `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (Secondary fallback)
  3. `nvidia/nemotron-3.5-lightning-30b-a3b` (Tertiary fallback)
- **Score Band Color Coding**:
  - `Strong` (≥ 80)
  - `Moderate` (60–79)
  - `Weak` (< 60)

---

## Architecture & Workflow

```
       User (Telegram)
              │
      Upload JD Document
              │
    Upload Resume Document(s)
              │
    Text Extraction (PDF / DOCX / TXT)
              │
    NVIDIA NIM API (Model Fallback Chain)
              │
    JSON Parsing & Defensive Clamping
              │
 ┌────────────┴────────────┐
 │                         │
Score & Gap Report    Score Comparison Chart
 (Telegram Message)     (Matplotlib Image)
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- Telegram Bot Token (from [@BotFather](https://t.me/botfather))
- NVIDIA NIM API Key (from [NVIDIA NGC / Build](https://build.nvidia.com/))

### 1. Clone the Repository

```bash
git clone https://github.com/Rishi1435/Resume-JD_ChatBot.git
cd Resume-JD_ChatBot
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory (or copy `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
NVIDIA_API_KEY=your_nvidia_api_key_here
```

### 5. Run the Bot

```bash
python main.py
```

---

## Bot Commands

- `/start` — Initialize or reset session. Prompts for a Job Description (JD).
- `/compare` — Generate an aligned comparison table of all resumes analyzed under the current JD.
- `/history` — List summary of analyzed resumes with timestamps and ATS scores.

---

## Running Unit Tests

Unit tests validate score clamping, course filtering, score-band assignment, and ranking logic:

```bash
python -m pytest test_filter.py -v
```

---

## Security

Sensitive keys and secrets are stored in `.env` and excluded from version control via `.gitignore`. Never commit API keys or bot tokens to the repository.
