# 🎓 UEL Enhanced AI Admission Assistant

## Overview

The **UEL Enhanced AI Admission Assistant** is a comprehensive, research-oriented AI system designed to demonstrate how multiple AI components can be **integrated into a single, coherent admission-support workflow**. Instead of showcasing isolated machine learning models, this project focuses on **system-level AI engineering**, where NLP, machine learning, analytics, persistence, and an interactive UI operate together.

This repository represents a **first integrated version** of the system, prioritising architectural clarity, modularity, and academic rigor over production-scale deployment.

> ⚠️ **Important Disclaimer**
> This is an **academic and personal project** created strictly for learning, research, and dissertation demonstration.
---

## 🎯 Motivation & Problem Context

University admission processes involve multiple decision points:

* Personalised AI chat bot availability 
* Course suitability assessment
* Applicant profile evaluation
* Interview readiness and performance
* Document verification
* Decision support and analytics

Most AI projects address **only one** of these problems in isolation. This project was created to explore how **multiple AI-driven components can coexist within a unified system**, sharing context, data, and evaluation logic — a scenario closer to real-world AI deployments.

---

## ✨ Core Capabilities

### 🧩 Profile-Driven Personalisation

* Persistent applicant profiles stored locally using SQLite
* Shared profile context across all modules
* System behaviour adapts based on:

  * Academic background
  * Skills and interests
  * Historical interactions
* Enables consistent, context-aware recommendations and evaluations

### 🧠 AI & Analytics Modules

#### 🎓 Course Recommendation

* Feature-based recommendation logic
* Uses applicant profile attributes
* Designed for extensibility (future collaborative filtering / deep models)

#### 🗣️ Interview Preparation & Evaluation

* Practice interview workflows
* Response relevance checking
* Sentiment analysis of textual responses
* Performance scoring using custom evaluation logic

#### 📄 Document Verification

* Rule-based structural checks
* NLP-assisted content consistency analysis
* Prototype-level verification for academic demonstration

#### 😊 Sentiment Analysis

* Transformer-based and classical NLP approaches
* Applied to interview responses and textual inputs

#### 📊 Predictive Analytics (Experimental)

* Exploratory models for admission outcome analysis
* Included to demonstrate analytics integration, not production decisions

---

## 💡 Key Strengths of This Project

* **All-in-One Integrated Prototype**: Multiple AI components operating within a shared workflow
* **System-Level Thinking**: Focus on architecture and data flow, not just model accuracy
* **Profile-Centric Design**: Persistent applicant context reused across modules
* **Modular Architecture**: Each component can be extended, replaced, or evaluated independently
* **Research-Oriented Implementation**: Suitable for MSc / PhD review and applied AI portfolios

This system demonstrates how AI solutions evolve from isolated experiments into **cohesive, extensible platforms**.

---

## 🧠 Design Decisions & Trade-offs

* SQLite was chosen over heavier databases to keep the system lightweight and reproducible.
* Streamlit was selected for rapid prototyping and academic demonstrations rather than production UI.
* Local LLM integration (Ollama) was kept optional to preserve privacy and offline execution.
* Some models are intentionally experimental to prioritise architectural exploration over optimisation.

---

## 🛠️ Tech Stack (Complete & Accurate)

### Core Technologies

* **Language**: Python 3.10+
* **Frontend / UI**: Streamlit
* **Database**: SQLite
* **Configuration Management**: INI-based configuration files (`config.ini`)

### Machine Learning & NLP

* **Classical ML**: scikit-learn
* **NLP**: transformers (BERT-style models)
* **Deep Learning**: TensorFlow / Keras (experimental components)

### Data Handling & Processing

* **Data Manipulation**: pandas
* **Numerical Computing**: NumPy

### LLM Integration (Optional, Local)

* **Local LLM Runtime**: Ollama
* Used only for selected components
* Not required for core system functionality

### Evaluation & Experimentation

* Custom evaluation pipelines
* sklearn-based metrics
* Experimental validation scripts

---

## 🚀 Running the System Locally

```bash
# 1. Clone the repository
git clone https://github.com/MuhammadAhmed-38/uel-enhanced-ai-assistant.git
cd uel-enhanced-ai-assistant

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit interface
streamlit run streamlit_interface.py
```

---

## ⚠️ Deployment & Environment Constraints

* This project is **not fully compatible with Streamlit Cloud**
* Reason: Certain components rely on **local resources**, including:

  * Local LLM inference via Ollama
  * File-based persistence and recordings

### Recommended Environments

* Local development machines
* Academic or private servers
* Controlled demonstration environments

---

## 📊 Evaluation & Validation

* Evaluation logic implemented in: `performance_evaluator.py`
* Metrics include:

  * Relevance scoring
  * Sentiment polarity
  * Basic accuracy measures where applicable
* Data sources:

  * Synthetic applicant profiles
  * Simulated interview responses

This project emphasises **evaluation transparency**, while keeping detailed experimental methodology outside the main README for clarity.

---

## 📂 Project Structure (High-Level)

```
uel-enhanced-ai-assistant/
│
├── streamlit_interface.py     # Main UI entry point
├── main.py                    # System orchestration
├── unified_uel_ai_system.py   # Integrated system logic
├── course_recommendation.py
├── interview_preparation.py
├── document_verification.py
├── sentiment_analysis.py
├── performance_evaluator.py
├── profile_manager.py
├── database_manager.py
├── utils.py
├── data/
├── tests/
├── recordings/
├── requirements.txt
└── README.md
```

---

## 📸 Result's Screenshots

### Admission Prediction Flowchart
![Admission Prediction Flowchart](screenshots/Admission_Prediction_Flowchart.jpg)

### Admission Prediction Accuracy
![Admission Prediction Accuracy](screenshots/Admission_Prediction_Accuracy.jpg)

### AI Chat Bot Accuracy and Feature Importance
![AI Chat Bot Accuracy and Feature Importance](screenshots/AI_Chat_Bot_Accuracy_and_Feature_Importance.jpg)

### AI Chat Bot Accuracy
![AI Chat Bot Accuracy](screenshots/AI_Chat_Bot_Accuracy.jpg)

### AI Chat Bot Personalised Answers
![AI Chat Bot Personalised Answers](screenshots/AI_Chat_Bot_Personalised_Answers.jpg)

### Basic Tier AI Chat Bot Interface
![Basic Tier AI Chat Bot Interface](screenshots/Basic_Tier_AI_Chat_Bot_Interface.jpg)

### Basic Tier AI Chat Bot Response
![AI Chat Bot Accuracy and Feature Importance](screenshots/Basic_Tier_AI_Chat_Bot_Response.jpg)

### Basic Tier Dashboard
![AI Chat Bot Accuracy](screenshots/Basic_Tier_Dashboard.jpg)

### Course Prediction Accuracy
![AI Chat Bot Accuracy](screenshots/Course_Prediction_Accuracy.jpg)

### Course Recommendation Accuracy
![AI Chat Bot Accuracy](screenshots/Course_Recommendation_Accuracy.jpg)

### Course Recommendation Confusion Matrix
![AI Chat Bot Accuracy](screenshots/Course_Recommendation_Confusion_Matrix.jpg)

### Course Recommendation Flowchart
![AI Chat Bot Accuracy](screenshots/Course_Recommendation_Flowchart.jpg)

### Document Verificatio Accuracy0
![AI Chat Bot Accuracy](screenshots/Document_Verificatio_Accuracy0.jpg)

### Document Verificatio Accuracy1
![AI Chat Bot Accuracy](screenshots/Document_Verification_Accuracy.jpg)

### Document Verificatio Flowchart
![AI Chat Bot Accuracy](screenshots/Document_Verification_Flowchart.jpg)

### Feature Importance Across ML Models
![AI Chat Bot Accuracy](screenshots/Feature_Importance_Across_ML_Models.jpg)

### Interview And Cheating Detection Accuracies And Confusion Matrix
![AI Chat Bot Accuracy](screenshots/Interview_And_Cheating_Detection_Accuracies_And_Confusion_Matrix.jpg)

### Interview Prep Flowchart
![AI Chat Bot Accuracy](screenshots/Interview_Preparation_Flowchart.jpg)

### Large Language Model Flowchart
![AI Chat Bot Accuracy](screenshots/LLM_Flowchart.jpg)

### ML Models Accuracies
![AI Chat Bot Accuracy](screenshots/ML_Models_Accuracies.jpg)

### Personalised Tier Dashboard0
![AI Chat Bot Accuracy](screenshots/Personalised_Tier_Dashboard0.jpg)

### Personalised Tier Dashboard1
![AI Chat Bot Accuracy](screenshots/Personalised_Tier_Dashboard1.jpg)

### Personalised Tier Dashboard2
![AI Chat Bot Accuracy](screenshots/Personalised_Tier_Dashboard2.jpg)

### Personalised Tier Dashboard3
![AI Chat Bot Accuracy](screenshots/Personalised_Tier_Dashboard3.jpg)

### Personalised Tier Dashboard4
![AI Chat Bot Accuracy](screenshots/Personalised_Tier_Dashboard4.jpg)

### Personalised Tier Dashboard Profile Complete
![AI Chat Bot Accuracy](screenshots/Personalised_Tier_Dashboard_Profile_Complete.jpg)

### Project Architecture
![AI Chat Bot Accuracy](screenshots/Project_Architecture.jpg)

### RAG Flowchart
![AI Chat Bot Accuracy](screenshots/RAG_Flowchart.jpg)

### Results
![AI Chat Bot Accuracy](screenshots/Results.jpg)

### System Architecture
![AI Chat Bot Accuracy](screenshots/System_Architecture.jpg)

### System Initialisation Flowchart
![AI Chat Bot Accuracy](screenshots/System_Initialisation_Flowchart.jpg)

### System Performance Accuracy Among ML Models
![AI Chat Bot Accuracy](screenshots/System_Performance_Accuracy_Among_ML_Models.jpg)

### Working Flowchart
![AI Chat Bot Accuracy](screenshots/Working_Flowchart.jpg)

---

## 🔭 Current Limitations & Future Work

**Current limitations:**

* Limited scale of data from faqs of 20 univeristies and admission staff feedback about 1GB size
* Local-only execution for LLM-based components
* Simplified evaluation metrics for certain modules

**Planned future work:**

* Larger and more diverse datasets
* Stronger quantitative benchmarking
* API layer for external system integration
* Optional cloud-compatible inference pathways

---

## 🎓 Intended Audience

* MSc / PhD evaluators
* AI researchers and applied ML engineers
* Recruiters assessing system-level AI skills
* Developers exploring integrated AI decision-support systems

---

## 📚 Citation & Usage

If you use this project for academic reference or inspiration, please cite or acknowledge the repository.

This project is available for freely used for educational and research purposes.

---

## 👤 Author

**Muhammad Ahmed**
MSc Artificial Intelligence (Distinction)

GitHub: [https://github.com/MuhammadAhmed-38](https://github.com/MuhammadAhmed-38)

---

## ⭐ Feedback & Contributions

Suggestions, issues, and academic discussions are welcome via GitHub Issues.
