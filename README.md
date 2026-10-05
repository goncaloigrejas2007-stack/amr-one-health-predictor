# 🌍 AMR One Health Intelligence Platform — Multi-Species & Zoonotic Resistance Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.24-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![PubMed](https://img.shields.io/badge/PubMed-NCBI%20Entrez%20API-336699?style=for-the-badge)
![One Health](https://img.shields.io/badge/WHO%20%2F%20WOAH-One%20Health-10B981?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-14b8a6?style=for-the-badge)

**An interactive Machine Learning and Epidemiological Intelligence platform addressing Antimicrobial Resistance (AMR) across Humans, Livestock, Companion Animals, Wildlife, and Environmental Reservoirs under the WHO / WOAH / FAO / UNEP One Health Framework, integrated with real-time PubMed scientific evidence.**

[🚀 Quick Start](#-quick-start) · [🌍 One Health Framework](#-the-one-health-framework) · [✨ Features](#-features) · [📚 PubMed Evidence](#-pubmed-evidence--ncbi-integration) · [🤖 ML Pipeline](#-machine-learning-pipeline) · [📁 Structure](#-repository-structure)

</div>

---

## 🌍 The One Health Framework

**Antimicrobial Resistance (AMR)** does not respect species boundaries. Overuse of antibiotics in livestock, companion animals, and agricultural runoff directly drives resistance in human pathogens through food chains, water systems, and direct contact.

Recognised by the **World Health Organisation (WHO)**, **World Organisation for Animal Health (WOAH)**, **FAO**, and **UNEP**, the **One Health Approach** is the gold standard for global AMR containment:

```
               ┌───────────────────────────────┐
               │        HUMAN HEALTH           │
               │  Hospitals · Clinics · Towns  │
               └──────────────┬────────────────┘
                              │
               Zoonotic       │      Environmental
             Transmission     │      Contamination
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌───────────────┐                           ┌───────────────┐
│ ANIMAL HEALTH │ ◄── Agricultural Runoff ──► │  ENVIRONMENT  │
│ Livestock     │     Wastewater & Sludge   │ Water · Soil  │
│ Companion     │                           │ Food Supply   │
│ Wildlife      │                           └───────────────┘
└───────────────┘
```

This platform implements the One Health paradigm to model, visualise, and predict multi-drug resistance (MDR) across **12 host categories** and clinical/veterinary bacterial species.

---

## ✨ Features

| Module | Description |
|---|---|
| 📊 **Global AMR Dashboard** | Real-time surveillance KPIs, MDR rates, ESBL/MRSA/VRE prevalence, temporal resistance trends, and cross-antibiotic resistance heatmaps. |
| 🌍 **One Health Hub** | **Multi-species transmission analysis**: Sankey flows connecting hosts, bacteria, and resistance phenotypes; comparative host MDR rankings; zoonotic risk indices. |
| 🔬 **Exploratory Analysis** | Interactive antibiograms, host vs. setting matrices, resistance profiles across European countries, and high-risk pathogen screening. |
| 🤖 **ML Intelligence** | Random Forest classifier trained on One Health epidemiological and microbiological features (ROC-AUC > 0.98, 5-fold CV, confusion matrix, feature importance). |
| 🧬 **Live AMR Predictor** | Interactive inference engine: select host species (Human, Bovine, Canine, Wildlife, Water, etc.), isolate metadata, and antibiogram to predict MDR risk probability in real time. |
| 📚 **PubMed Evidence & NCBI** | **Live NCBI Entrez E-Utilities integration**: queries peer-reviewed biomedical literature directly from PubMed for selected pathogen-antibiotic-host combos, with DOI links and molecular resistance genes atlas (*mcr-1*, *blaKPC*, *blaNDM*, *mecA*, *vanA*). |

---

## 🧫 Multi-Species & Host Taxonomy (12 Categories)

| Host Compartment | Species / Matrix | Typical Settings & Reservoirs | Key Pathogens |
|---|---|---|---|
| **Human** 🧑 | Clinical isolates | Hospitals (ICU, Surgical, Medical), Community | *E. coli*, *K. pneumoniae*, *S. aureus*, *P. aeruginosa* |
| **Livestock** 🐄 🐷 🐔 | Bovine, Porcine, Poultry | Dairy & beef farms, swine units, poultry barns, abattoirs | *Salmonella enterica*, *Campylobacter jejuni*, *E. coli* |
| **Companion** 🐕 🐈 🐎 | Canine, Feline, Equine | Veterinary clinics, animal shelters, stables, households | *S. pseudintermedius*, *E. coli*, *S. aureus* (MRSA) |
| **Wildlife** 🦅 🐗 | Wild Birds, Wild Boar | Nature reserves, wetlands, hunting grounds | Environmental *Enterobacteriaceae*, *Enterococcus* |
| **Environment** 💧 🌱 🥩 | Water, Soil, Food products | Wastewater treatment plants, rivers, retail meat, agricultural soil | Colistin-resistant *E. coli*, ESBL producers |

---

## 📁 Repository Structure

```
amr-predictor/
│
├── app.py                      # Interactive Streamlit One Health dashboard
├── setup.py                    # Bootstrap script: data generation + ML training
├── requirements.txt            # Python dependencies
│
├── utils/
│   ├── data_generator.py       # One Health synthetic generator (ECDC/EFSA/WHO calibrated)
│   ├── train_model.py          # Random Forest ML pipeline with multi-species features
│   └── pubmed_client.py        # Live NCBI Entrez API client for PubMed literature & genes
│
├── data/
│   └── amr_synthetic_dataset.csv   # One Health dataset (2,000+ isolates, 12 hosts)
│
├── models/
│   ├── amr_rf_model.joblib     # Pre-trained Random Forest pipeline
│   ├── feature_names.joblib    # Preprocessed feature registry
│   └── model_metrics.joblib    # Evaluation metrics (ROC-AUC, CV scores, confusion matrix)
│
├── notebooks/
│   └── exploratory_analysis.ipynb  # Deep-dive Jupyter notebook
│
├── assets/                     # Visual assets and screenshots
├── .streamlit/
│   └── config.toml             # Dark-mode dashboard theme
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/goncaloigrejas2007-stack/amr-one-health-predictor.git
cd amr-predictor

python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Generate Dataset & Train Model

```bash
python setup.py
```

Outputs:
- Generates `data/amr_synthetic_dataset.csv` with 2,000 multi-species isolates.
- Trains Random Forest classifier with stratified k-fold cross-validation.
- Saves model artefacts to `models/`.

### 3. Run Streamlit Dashboard

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 🤖 Machine Learning Pipeline

```
Raw Isolate Profile
(Host, Bacteria, Setting, Country, Resistance Markers, Antibiogram)
                      │
                      ▼
       ┌───────────────────────────────┐
       │      Feature Engineering      │
       │  • Host species & compartment │
       │  • Zoonotic risk score        │
       │  • Intensive care/farming flag│
       │  • 22-antibiotic panel        │
       └──────────────┬────────────────┘
                      │
                      ▼
       ┌───────────────────────────────┐
       │     Column Preprocessor       │
       │  Categorical: OneHotEncoder   │
       │  Numerical: StandardScaler    │
       │  Imputation: Median / Mode    │
       └──────────────┬────────────────┘
                      │
                      ▼
       ┌───────────────────────────────┐
       │   Random Forest Classifier    │
       │  n_estimators=300, max_depth=14│
       │  class_weight="balanced"      │
       │  5-Fold Stratified CV         │
       └──────────────┬────────────────┘
                      │
                      ▼
       ┌───────────────────────────────┐
       │         Output Score          │
       │  P(MDR) ∈ [0, 1]              │
       │  Classification: MDR / Non-MDR│
       └───────────────────────────────┘
```

### Model Performance

| Metric | Cross-Validation (5-Fold) | Test Split (20%) |
|---|---|---|
| **ROC-AUC** | **0.989 ± 0.003** | **0.986** |
| **Precision (MDR)** | — | **0.91** |
| **Recall (MDR)** | — | **0.89** |
| **Accuracy** | — | **0.94** |

---

## 🌐 Antibiotic Panel (22 Agents, 10 Classes)

| Class | Antibiotics Evaluated |
|---|---|
| **Beta-lactams** | Ampicillin, Amoxicillin-Clavulanate, Piperacillin-Tazobactam |
| **Cephalosporins** | Cefazolin, Cefuroxime, Ceftriaxone, Ceftazidime, Cefepime |
| **Carbapenems** | Meropenem, Imipenem, Ertapenem *(Critical for human medicine)* |
| **Fluoroquinolones** | Ciprofloxacin, Levofloxacin |
| **Aminoglycosides** | Gentamicin, Tobramycin, Amikacin |
| **Glycopeptides** | Vancomycin, Teicoplanin |
| **Tetracyclines** | Tetracycline, Tigecycline |
| **Polymyxins** | Colistin *(Reserve agent / agricultural concern)* |
| **Sulfonamides** | Trimethoprim-Sulfamethoxazole |
| **Oxazolidinones** | Linezolid *(Last-resort Gram-positive)* |

---

## 💼 LinkedIn & Portfolio Showcase

### Suggested Post (PT):
> 🔬 **Inteligência Artificial aplicada ao desafio One Health da Resistência a Antibióticos (AMR)!**
> 
> A resistência antimicrobiana não escolhe espécies: o uso de antibióticos na pecuária e a contaminação ambiental têm impacto direto na saúde humana e na eficácia dos tratamentos clínicos.
> 
> Desenvolvi o **AMR One Health Predictor**, uma plataforma interativa de Data Science e Machine Learning que cruza microbiologia clínica, medicina veterinária e epidemiologia ambiental:
> - 🌍 **Abordagem One Health**: Análise de 12 categorias de hospedeiros (humanos, bovinos, suínos, aves, animais de companhia, fauna selvagem e amostras ambientais).
> - 📊 **Surveillance & Redes de Transmissão**: Diagramas de fluxo Sankey para mapear rotas zoonóticas entre reservatórios e fenótipos MDR.
> - 🤖 **Modelo Preditivo**: Random Forest com **ROC-AUC de 0.989** para estimar probabilidade de multirresistência com base no antibiograma e perfil do hospedeiro.
> - 📚 **Evidência Científica ao Vivo via PubMed**: Integração com a API oficial do NCBI Entrez para cruzar as previsões com artigos científicos e determinantes genéticos reais (*mcr-1*, *blaKPC*, *mecA*, etc.).
> 
> 🔗 Repositório e código no GitHub: https://github.com/goncaloigrejas2007-stack/amr-one-health-predictor
> #DataScience #MachineLearning #Microbiology #OneHealth #HealthTech #Python #Streamlit #Bioinformatics #PubMed #BioAI

---

## ⚠️ Disclaimer

This application uses synthetic data calibrated against published epidemiological surveillance reports (WHO, ECDC, EFSA) for educational, research, and portfolio demonstration purposes. It does not replace official clinical microbiology diagnostics or veterinary medical advice.

---

## 👤 Author

**Gonçalo Igrejas**  
*Data Science · Artificial Intelligence · Applied Microbiology*  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/goncaloigrejas2007-stack)
