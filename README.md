# DutySecure: Privacy-First AI Roster Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Model](https://img.shields.io/badge/Model-Llama--3.2--1B-orange?style=flat-square)
![Privacy](https://img.shields.io/badge/Privacy-Offline-green?style=flat-square)



## Project Overview

I originally developed this tool as a personal initiative to assist my father, who manages complex staff rosters. He was spending significant time daily manually parsing Excel sheets, searching for specific dates, and filtering for shifts to allocate duties.

**Evolution of the Software:**
* **Version Legacy:** Relied on hard-coded string matching and manual column search. It was functional but fragile; changes to column headers (e.g., renaming "Name" to "Staff List") would break the execution.
* **Version 1.0 (Current):** I re-engineered the core architecture to address structural variability. The system now integrates a **Local Large Language Model (LLM)**. It parses spreadsheet headers semantically to understand context, identifying that terms like "Resource", "Employee", and "Staff Name" represent the same entity without requiring manual reconfiguration.

---

## Key Features

### Local Intelligence (Privacy-Focused)
Unlike cloud-based AI tools, DutySecure processes data entirely offline. This is critical for handling rosters containing Personally Identifiable Information (PII).
* **Engine:** Powered by `Llama-3.2-1B` (Quantized).
* **Privacy:** All inference occurs on the local CPU. No data is transmitted to external servers.

### Smart Semantic Search
The application replaces rigid exact-match logic with a hybrid search approach:
* **Fuzzy Matching:** Handles typographical errors in headers.
* **LLM Reasoning:** Analyzes ambiguous column names to deduce their function (e.g., identifying the correct "Date" or "Shift Code" column).

### Modern User Interface
Transformed from a command-line script to a full-stack web application using Flask. The UI features a dark-mode, responsive design to improve accessibility for non-technical users.

---

## Technical Architecture

* **Core Framework:** Python 3.10, Flask
* **AI & Inference:** `llama-cpp-python` (GGUF model execution), HuggingFace Hub
* **Data Processing:** Pandas, OpenPyXL, RapidFuzz
* **Build & Distribution:** PyInstaller (Custom build pipeline implementing smart path handling for persistent model storage)

---

## Workflow

1.  **Ingestion:** The user uploads a raw `.xlsx` roster file.
2.  **Sanitization:** The algorithm scans initial rows to identify the actual header row, bypassing metadata or logos common in corporate reports.
3.  **Analysis:** The Local LLM evaluates column headers.
    -> *Query:* "Identify the column representing the shift code."
    -> *Inference:* "Column 'Allocated_Time' is the shift column; 'Emp_ID' is the identifier."
4.  **Extraction:** The system filters the dataset based on the user's selected date and shift parameters.

---

## Installation & Usage

**Note:** This application is compiled as a standalone executable for Windows. Python installation is not required.

1.  Navigate to the **Releases** page of this repository.
2.  Download `DutySecure.exe`.
3.  Execute the application.
    * *First Run:* The application will automatically download the AI Model (~1GB) to a local folder named `DutySecure_Models`.
    * *Subsequent Runs:* The application detects the local model and launches instantly.

---

## Developer

**Designed & Developed by [Avnish (24KaratAu)](https://github.com/24KaratAu)**
