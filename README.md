# EduMetric AI: PDF-to-MCQ Comparative Assessment System

A comprehensive, research-grade PDF-to-MCQ question generation and multi-model comparative evaluation system. This application extracts study material from uploaded PDFs, executes four distinct model pipelines under identical assessment constraints, delivers a blind test to students, and computes 6 standardized mathematical metrics evaluated against an 8-check deterministic quality gate.

---

## 🚀 Key Features

1. **Four Models Compared Fairly:**
   - **Model 1 (My Model):** Advanced Multi-Stage RAG Pipeline (Page-Aware Segmentation $\rightarrow$ Semantic Chunking $\rightarrow$ Dense Vector Index $\rightarrow$ Factual Blueprint Extraction $\rightarrow$ Context Retrieval $\rightarrow$ MCQ Generation $\rightarrow$ Distractor Validation $\rightarrow$ Answer Verification $\rightarrow$ Semantic Deduplication $\rightarrow$ Quality Gate Filtering).
   - **Model 2 (Qwen 2.5 3B):** `Qwen/Qwen2.5-3B-Instruct` Baseline.
   - **Model 3 (Phi-3.5 Mini):** `microsoft/Phi-3.5-mini-instruct` Baseline.
   - **Model 4 (Mistral 7B 4-bit):** `mistralai/Mistral-7B-Instruct-v0.2` with BitsAndBytes 4-bit NF4 Quantization Baseline.

2. **Objective 8-Check Deterministic Quality Gate:**
   - **Check 1 (Grounding):** Source PDF textual and conceptual support verification.
   - **Check 2 (Answer Validity):** Verifiable correctness of designated answer in context.
   - **Check 3 (Option Count):** Exactly 4 options ($A, B, C, D$).
   - **Check 4 (Single Correct Key):** Exactly one distinct correct answer option.
   - **Check 5 (Distractor Quality):** Domain-relevant, non-empty, and distinct distractors.
   - **Check 6 (Non-Duplication):** Semantic similarity threshold enforcement ($\text{similarity} < 0.85$).
   - **Check 7 (Clarity & Grammar):** Well-formed phrasing and formatting syntax validation.
   - **Check 8 (No Hallucination):** No unsupported external knowledge assumptions.

3. **Scientifically Calculated Metrics (No Fabricated Values):**
   - **Accuracy** = $\frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$
   - **Precision** = $\frac{\text{TP}}{\text{TP} + \text{FP}}$
   - **Recall** = $\frac{\text{TP}}{\text{TP} + \text{FN}}$
   - **F1 Score** = $\frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
   - **False Negative Rate (FNR)** = $\frac{\text{FN}}{\text{FN} + \text{TP}}$
   - **False Positive Rate (FPR)** = $\frac{\text{FP}}{\text{FP} + \text{TN}}$
   *(Safe handling for zero denominators included)*

4. **Modern Light-Mode Student Assessment Portal:**
   - Student Login + 2-Step OTP Verification.
   - Configurable questions ($5, 10, 15, 20, 25, 30$), time limits ($5 - 60$ mins), and difficulty distributions.
   - **Blind Assessment:** Students take the assessment using My Model MCQs without seeing model identifiers, metrics, or answers during the test.
   - Post-test review reveals student score, correct answers, explanations, and exact PDF page source evidence.

5. **Automated Multi-Format Artifact Outputs:**
   - Unique output folder created per run: `outputs/<clean_pdf_name>_<timestamp>/`.
   - **6 Matplotlib Comparison Bar Graphs:** `accuracy.png`, `precision.png`, `recall.png`, `f1_score.png`, `fnr.png`, `fpr.png`.
   - **Excel Report Workbook:** `<clean_pdf_name>_metrics.xlsx` with 9 individual sheets.
   - **Formal PDF Report:** `experiment_report.pdf` generated with ReportLab.
   - **JSON / CSV Exports:** `metrics.json`, `metrics.csv`, `generated_questions.json`, `student_results.json`.

---

## 📁 Project Structure

```
mini pro/
│
├── app.py                     # Main Streamlit Light-Mode Application
├── requirements.txt           # Python Dependencies
├── README.md                  # System Documentation
│
├── models/
│   ├── __init__.py
│   └── model_pipeline.py     # RAG, Vector Search, Blueprinting, and Baseline Pipelines
│
├── utils/
│   ├── __init__.py
│   └── evaluation.py         # 8-Check Quality Gate, Metrics, 6 Graphs, Excel & PDF Generators
│
└── outputs/                   # Automated Assessment & Experiment Artifact Storage
    └── <PDF_NAME>_<TIMESTAMP>/
        ├── accuracy.png
        ├── precision.png
        ├── recall.png
        ├── f1_score.png
        ├── fnr.png
        ├── fpr.png
        ├── experiment_report.pdf
        ├── <PDF_NAME>_metrics.xlsx
        ├── metrics.json
        ├── metrics.csv
        ├── generated_questions.json
        └── student_results.json
```

---

## 🛠️ Installation & Setup

1. **Clone or Navigate to the Workspace:**
   ```bash
   cd "c:/Users/Asus/Desktop/mini pro"
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit Portal:**
   ```bash
   streamlit run app.py
   ```

4. **Access the Portal:**
   Open `http://localhost:8501` in your browser.
   - **Student ID:** `student@university.edu` (or custom)
   - **Password:** Any password
   - **OTP:** Use the displayed Demo OTP code (e.g. `849201`) to complete verification.

---

## 📊 Evaluation & Experiment Workflow

1. Upload your course study material PDF in the **Assessment** tab.
2. Select desired question count, time limit, and difficulty.
3. Click **Start Assessment & Comparative Run**.
4. The system executes all 4 models sequentially, evaluates all candidate questions under the common 8-check deterministic quality gate, and prepares the blind student assessment.
5. Answer the questions and click **Submit Test**.
6. Review your score, examine the PDF source evidence, inspect the 4-model comparative metrics table, view the 6 comparison bar charts, and download the full PDF and Excel reports.
