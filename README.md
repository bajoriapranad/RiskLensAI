# RiskLens AI

## Live Demo

🚀 Demo: https://risklensai-pranad-bajoria.streamlit.app/


### AI-Powered Private Equity Due Diligence Copilot

RiskLens AI is an end-to-end financial risk assessment and due diligence platform designed to assist private equity analysts, investment teams, and financial professionals in evaluating company risk faster and more consistently.

The platform combines machine learning, explainable AI, financial ratio analysis, and automated report generation to transform raw financial statements into actionable investment insights.

---

## The Problem

Private equity firms, investment funds, and corporate finance teams review hundreds of companies before making investment decisions.

Traditional due diligence is:

* Time-consuming
* Highly manual
* Dependent on analyst experience
* Difficult to standardize across teams

Analysts spend significant effort:

* Calculating financial ratios
* Identifying risk drivers
* Creating investment memos
* Preparing due diligence questionnaires
* Building reports for investment committees

RiskLens AI was built to reduce this effort by acting as an intelligent due diligence copilot rather than replacing analysts.

---

## Solution

RiskLens AI automatically:

1. Ingests company financial data
2. Computes financial health indicators
3. Predicts bankruptcy risk using machine learning
4. Explains risk drivers using SHAP explainability
5. Generates due diligence questions
6. Creates investment-style memos
7. Exports analyst-ready PDF reports

---

# Product Workflow

Financial Statements (Excel)
↓
Financial Ratio Engine
↓
Feature Engineering
↓
XGBoost Risk Prediction
↓
SHAP Explainability
↓
Due Diligence Question Generation
↓
Investment Memo Generation
↓
PDF Report Export

---

# Key Features

## Bankruptcy Risk Prediction

Predicts probability of financial distress using a trained XGBoost model.

Outputs:

* Risk Score
* Bankruptcy Probability
* Risk Category
* Confidence Metrics

---

## Explainable AI (SHAP)

Unlike traditional black-box models, RiskLens explains:

* Why a company is risky
* Which ratios increase risk
* Which ratios reduce risk
* Relative contribution of each financial metric

Visualizations include:

* SHAP Waterfall Chart
* Feature Importance Chart
* Top Risk Drivers
* Top Risk Mitigants

---

## Due Diligence Question Generator

Automatically generates analyst-focused questions based on identified risk signals.

Examples:

* Why has leverage increased?
* Explain deterioration in working capital.
* What caused declining profitability?
* Clarify recent changes in liquidity position.

---

## Investment Memo Generator

Transforms model findings into a structured investment summary suitable for:

* Investment Committees
* Deal Teams
* Internal Reviews
* Academic Demonstrations

---

## PDF Report Export

Creates a complete downloadable report containing:

* Company Overview
* Risk Assessment
* SHAP Analysis
* Due Diligence Questions
* Investment Recommendation

---

## Excel-Based Analyst Workflow

RiskLens supports a realistic analyst workflow.

Analysts:

1. Download a standardized Excel template
2. Enter financial statement line items
3. Upload the completed workbook
4. Receive a complete due diligence assessment

No manual ratio calculations required.

---

# Machine Learning Pipeline

## Dataset

Polish Bankruptcy Prediction Dataset (3-Year Horizon)

* 10,503 companies
* 64 financial ratios
* Bankruptcy prediction target
* Highly imbalanced classification problem

---

## Data Processing

Implemented:

* Missing value handling
* Missing indicators
* Outlier treatment
* Log transformations
* Correlation reduction
* Financial feature engineering

---

## Feature Engineering

Additional engineered features include:

### Altman Z-Score

Classic financial distress indicator.

### Risk Flags

* Negative ROA
* High Debt Ratio
* Insolvency Flag
* Negative Working Capital
* Negative EBIT
* Negative Margin
* Cost Ratio > 1

### Missing Value Indicators

Binary features identifying unavailable financial information.

---

## Models Evaluated

### Logistic Regression

Baseline interpretable model.

### Random Forest

Non-linear ensemble benchmark.

### XGBoost

Selected as the final production model.

---

# Model Performance

| Metric    | Score  |
| --------- | ------ |
| ROC-AUC   | 0.9344 |
| PR-AUC    | 0.6349 |
| Precision | 0.6812 |
| Recall    | 0.4747 |
| F1 Score  | 0.5595 |

Final Model: XGBoost

---

# Technology Stack

Frontend:

* Streamlit

Machine Learning:

* Scikit-Learn
* XGBoost

Explainability:

* SHAP

Data Processing:

* Pandas
* NumPy
* SciPy

Visualization:

* Plotly
* Matplotlib

Excel Integration:

* OpenPyXL

Reporting:

* ReportLab

Version Control:

* Git
* GitHub

---

# Unique Selling Proposition (USP)

Most student projects stop at:

Dataset → Model → Accuracy

RiskLens AI goes further:

Financial Statements
↓
Risk Prediction
↓
Explainability
↓
Due Diligence Questions
↓
Investment Memo
↓
PDF Report

The project focuses on supporting analysts rather than replacing them.

RiskLens AI transforms machine learning predictions into actionable investment intelligence.

---

# Future Enhancements

* Indian market-specific training data
* NSE/BSE financial statement integration
* Real-time company screening
* Portfolio-level risk monitoring
* LLM-powered due diligence assistant
* Deal sourcing and screening engine
* Multi-company comparison dashboard

---

# Author

Pranad Bajoria

B.Tech Computer Science & Engineering (AI & ML)

VIT University

Passionate about AI, Machine Learning, Financial Analytics, and Building Real-World Intelligent Systems.
