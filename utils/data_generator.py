"""
data_generator.py  (One Health Edition)
-----------------------------------------
Generates a realistic synthetic One Health AMR dataset covering:
  • Humans         (clinical isolates)
  • Livestock      (bovine, porcine, poultry)
  • Companion      (canine, feline, equine)
  • Wildlife       (wild birds, wild boar)
  • Environmental  (water, soil, food)

Bacterial species, resistance rates and zoonotic links are calibrated
to ECDC/EFSA/WHO One Health AMR reports (2018-2024).

Author : Gonçalo Igrejas
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(seed=42)

# ─── Antibiotic panel ─────────────────────────────────────────────────────────
ANTIBIOTIC_CLASSES = {
    "Beta-lactams":     ["Ampicillin", "Amoxicillin-Clavulanate", "Piperacillin-Tazobactam"],
    "Cephalosporins":   ["Cefazolin", "Cefuroxime", "Ceftriaxone", "Ceftazidime", "Cefepime"],
    "Carbapenems":      ["Meropenem", "Imipenem", "Ertapenem"],
    "Fluoroquinolones": ["Ciprofloxacin", "Levofloxacin"],
    "Aminoglycosides":  ["Gentamicin", "Tobramycin", "Amikacin"],
    "Glycopeptides":    ["Vancomycin", "Teicoplanin"],
    "Tetracyclines":    ["Tetracycline", "Tigecycline"],
    "Polymyxins":       ["Colistin"],
    "Sulfonamides":     ["Trimethoprim-Sulfamethoxazole"],
    "Oxazolidinones":   ["Linezolid"],
}
ALL_ANTIBIOTICS = [ab for lst in ANTIBIOTIC_CLASSES.values() for ab in lst]

# ─── Host categories ──────────────────────────────────────────────────────────
HOST_CATEGORIES = {
    "Human":       {"type": "Human",       "icon": "🧑", "setting": ["Hospital", "Community", "Long-term care"]},
    "Bovine":      {"type": "Livestock",   "icon": "🐄", "setting": ["Dairy farm", "Beef farm", "Veterinary clinic"]},
    "Porcine":     {"type": "Livestock",   "icon": "🐷", "setting": ["Pig farm", "Abattoir", "Veterinary clinic"]},
    "Poultry":     {"type": "Livestock",   "icon": "🐔", "setting": ["Broiler farm", "Layer farm", "Abattoir"]},
    "Canine":      {"type": "Companion",   "icon": "🐕", "setting": ["Veterinary clinic", "Animal shelter", "Home"]},
    "Feline":      {"type": "Companion",   "icon": "🐈", "setting": ["Veterinary clinic", "Animal shelter", "Home"]},
    "Equine":      {"type": "Companion",   "icon": "🐎", "setting": ["Equine clinic", "Racing stable", "Farm"]},
    "Wild Bird":   {"type": "Wildlife",    "icon": "🦅", "setting": ["Wildlife rescue", "Research station", "Nature reserve"]},
    "Wild Boar":   {"type": "Wildlife",    "icon": "🐗", "setting": ["Research station", "Hunting area", "Nature reserve"]},
    "Water":       {"type": "Environment", "icon": "💧", "setting": ["River", "Wastewater treatment", "Drinking water"]},
    "Soil":        {"type": "Environment", "icon": "🌱", "setting": ["Agricultural field", "Urban soil", "Forest"]},
    "Food":        {"type": "Environment", "icon": "🥩", "setting": ["Slaughterhouse", "Retail", "Processing plant"]},
}

HOST_WEIGHTS = {
    "Human": 0.28, "Bovine": 0.10, "Porcine": 0.10, "Poultry": 0.10,
    "Canine": 0.08, "Feline": 0.06, "Equine": 0.04,
    "Wild Bird": 0.06, "Wild Boar": 0.04,
    "Water": 0.06, "Soil": 0.04, "Food": 0.04,
}

# ─── Bacteria × Host compatibility ────────────────────────────────────────────
BACTERIA = {
    # Zoonotic generalists (infect multiple hosts)
    "Escherichia coli": {
        "gram": "Negative", "family": "Enterobacteriaceae",
        "hosts": ["Human","Bovine","Porcine","Poultry","Canine","Feline","Equine","Wild Bird","Wild Boar","Water","Soil","Food"],
        "zoonotic": True, "zoonotic_risk": "High",
        "esbl_rates": {"Human":0.28,"Bovine":0.15,"Porcine":0.20,"Poultry":0.25,"Canine":0.18,"Feline":0.15,"Equine":0.10,"Wild Bird":0.12,"Wild Boar":0.18,"Water":0.22,"Soil":0.10,"Food":0.20},
    },
    "Salmonella spp.": {
        "gram": "Negative", "family": "Enterobacteriaceae",
        "hosts": ["Human","Bovine","Porcine","Poultry","Wild Bird","Wild Boar","Food","Water"],
        "zoonotic": True, "zoonotic_risk": "High",
        "esbl_rates": {"Human":0.10,"Bovine":0.08,"Porcine":0.12,"Poultry":0.15,"Wild Bird":0.08,"Wild Boar":0.10,"Food":0.12,"Water":0.08},
    },
    "Campylobacter jejuni": {
        "gram": "Negative", "family": "Campylobacteraceae",
        "hosts": ["Human","Bovine","Poultry","Canine","Wild Bird","Food"],
        "zoonotic": True, "zoonotic_risk": "Very High",
        "esbl_rates": {"Human":0.0,"Bovine":0.0,"Poultry":0.0,"Canine":0.0,"Wild Bird":0.0,"Food":0.0},
    },
    "Klebsiella pneumoniae": {
        "gram": "Negative", "family": "Enterobacteriaceae",
        "hosts": ["Human","Bovine","Equine","Canine","Feline"],
        "zoonotic": True, "zoonotic_risk": "Moderate",
        "esbl_rates": {"Human":0.35,"Bovine":0.12,"Equine":0.10,"Canine":0.14,"Feline":0.12},
    },
    "Staphylococcus aureus": {
        "gram": "Positive", "family": "Staphylococcaceae",
        "hosts": ["Human","Bovine","Porcine","Canine","Feline","Equine","Poultry"],
        "zoonotic": True, "zoonotic_risk": "Moderate",
        "esbl_rates": {"Human":0.0,"Bovine":0.0,"Porcine":0.0,"Canine":0.0,"Feline":0.0,"Equine":0.0,"Poultry":0.0},
    },
    "Staphylococcus pseudintermedius": {
        "gram": "Positive", "family": "Staphylococcaceae",
        "hosts": ["Canine","Feline","Human"],
        "zoonotic": True, "zoonotic_risk": "Low",
        "esbl_rates": {"Canine":0.0,"Feline":0.0,"Human":0.0},
    },
    "Pseudomonas aeruginosa": {
        "gram": "Negative", "family": "Pseudomonadaceae",
        "hosts": ["Human","Canine","Feline","Equine","Water"],
        "zoonotic": False, "zoonotic_risk": "Low",
        "esbl_rates": {"Human":0.0,"Canine":0.0,"Feline":0.0,"Equine":0.0,"Water":0.0},
    },
    "Acinetobacter baumannii": {
        "gram": "Negative", "family": "Moraxellaceae",
        "hosts": ["Human","Canine","Equine","Soil"],
        "zoonotic": False, "zoonotic_risk": "Low",
        "esbl_rates": {"Human":0.0,"Canine":0.0,"Equine":0.0,"Soil":0.0},
    },
    "Enterococcus faecium": {
        "gram": "Positive", "family": "Enterococcaceae",
        "hosts": ["Human","Porcine","Poultry","Bovine","Canine","Food"],
        "zoonotic": True, "zoonotic_risk": "Moderate",
        "esbl_rates": {"Human":0.0,"Porcine":0.0,"Poultry":0.0,"Bovine":0.0,"Canine":0.0,"Food":0.0},
    },
    "Enterococcus faecalis": {
        "gram": "Positive", "family": "Enterococcaceae",
        "hosts": ["Human","Bovine","Porcine","Poultry","Canine","Feline"],
        "zoonotic": True, "zoonotic_risk": "Low",
        "esbl_rates": {"Human":0.0,"Bovine":0.0,"Porcine":0.0,"Poultry":0.0,"Canine":0.0,"Feline":0.0},
    },
    "Mannheimia haemolytica": {
        "gram": "Negative", "family": "Pasteurellaceae",
        "hosts": ["Bovine","Equine"],
        "zoonotic": False, "zoonotic_risk": "Negligible",
        "esbl_rates": {"Bovine":0.0,"Equine":0.0},
    },
    "Pasteurella multocida": {
        "gram": "Negative", "family": "Pasteurellaceae",
        "hosts": ["Bovine","Porcine","Poultry","Canine","Feline","Human"],
        "zoonotic": True, "zoonotic_risk": "Moderate",
        "esbl_rates": {"Bovine":0.0,"Porcine":0.0,"Poultry":0.0,"Canine":0.0,"Feline":0.0,"Human":0.0},
    },
    "Enterobacter cloacae": {
        "gram": "Negative", "family": "Enterobacteriaceae",
        "hosts": ["Human","Water","Soil"],
        "zoonotic": False, "zoonotic_risk": "Low",
        "esbl_rates": {"Human":0.22,"Water":0.15,"Soil":0.10},
    },
    "Clostridioides difficile": {
        "gram": "Positive", "family": "Clostridiaceae",
        "hosts": ["Human","Porcine","Canine","Wild Boar"],
        "zoonotic": True, "zoonotic_risk": "Moderate",
        "esbl_rates": {"Human":0.0,"Porcine":0.0,"Canine":0.0,"Wild Boar":0.0},
    },
}

# ─── Resistance probability tables (by species × antibiotic) ──────────────────
# Base rates from ECDC/EFSA reports; adjusted per host in generate_dataset()
RESISTANCE_PROBS_BASE = {
    "Escherichia coli": {
        "Ampicillin":0.55,"Amoxicillin-Clavulanate":0.30,"Piperacillin-Tazobactam":0.18,
        "Cefazolin":0.28,"Cefuroxime":0.25,"Ceftriaxone":0.20,"Ceftazidime":0.15,"Cefepime":0.15,
        "Meropenem":0.03,"Imipenem":0.03,"Ertapenem":0.05,
        "Ciprofloxacin":0.30,"Levofloxacin":0.28,
        "Gentamicin":0.18,"Tobramycin":0.15,"Amikacin":0.05,
        "Vancomycin":0.00,"Teicoplanin":0.00,
        "Tetracycline":0.40,"Tigecycline":0.05,
        "Colistin":0.02,"Trimethoprim-Sulfamethoxazole":0.35,"Linezolid":0.00,
    },
    "Salmonella spp.": {
        "Ampicillin":0.38,"Amoxicillin-Clavulanate":0.20,"Piperacillin-Tazobactam":0.12,
        "Cefazolin":0.15,"Cefuroxime":0.14,"Ceftriaxone":0.12,"Ceftazidime":0.10,"Cefepime":0.10,
        "Meropenem":0.01,"Imipenem":0.01,"Ertapenem":0.02,
        "Ciprofloxacin":0.25,"Levofloxacin":0.22,
        "Gentamicin":0.15,"Tobramycin":0.12,"Amikacin":0.04,
        "Vancomycin":0.00,"Teicoplanin":0.00,
        "Tetracycline":0.48,"Tigecycline":0.03,
        "Colistin":0.02,"Trimethoprim-Sulfamethoxazole":0.38,"Linezolid":0.00,
    },
    "Campylobacter jejuni": {
        "Ampicillin":0.30,"Amoxicillin-Clavulanate":0.25,"Piperacillin-Tazobactam":0.20,
        "Cefazolin":0.80,"Cefuroxime":0.80,"Ceftriaxone":0.12,"Ceftazidime":0.80,"Cefepime":0.80,
        "Meropenem":0.05,"Imipenem":0.05,"Ertapenem":0.05,
        "Ciprofloxacin":0.48,"Levofloxacin":0.48,
        "Gentamicin":0.05,"Tobramycin":0.05,"Amikacin":0.03,
        "Vancomycin":0.00,"Teicoplanin":0.00,
        "Tetracycline":0.55,"Tigecycline":0.02,
        "Colistin":0.00,"Trimethoprim-Sulfamethoxazole":0.32,"Linezolid":0.00,
    },
    "Klebsiella pneumoniae": {
        "Ampicillin":0.95,"Amoxicillin-Clavulanate":0.40,"Piperacillin-Tazobactam":0.30,
        "Cefazolin":0.35,"Cefuroxime":0.30,"Ceftriaxone":0.30,"Ceftazidime":0.25,"Cefepime":0.25,
        "Meropenem":0.10,"Imipenem":0.10,"Ertapenem":0.12,
        "Ciprofloxacin":0.30,"Levofloxacin":0.28,
        "Gentamicin":0.22,"Tobramycin":0.20,"Amikacin":0.08,
        "Vancomycin":0.00,"Teicoplanin":0.00,
        "Tetracycline":0.50,"Tigecycline":0.08,
        "Colistin":0.05,"Trimethoprim-Sulfamethoxazole":0.38,"Linezolid":0.00,
    },
    "Staphylococcus aureus": {
        "Ampicillin":0.70,"Amoxicillin-Clavulanate":0.20,"Piperacillin-Tazobactam":0.15,
        "Cefazolin":0.20,"Cefuroxime":0.20,"Ceftriaxone":0.20,"Ceftazidime":0.30,"Cefepime":0.25,
        "Meropenem":0.05,"Imipenem":0.05,"Ertapenem":0.05,
        "Ciprofloxacin":0.25,"Levofloxacin":0.22,
        "Gentamicin":0.20,"Tobramycin":0.18,"Amikacin":0.10,
        "Vancomycin":0.01,"Teicoplanin":0.01,
        "Tetracycline":0.30,"Tigecycline":0.02,
        "Colistin":1.00,"Trimethoprim-Sulfamethoxazole":0.20,"Linezolid":0.02,
    },
    "Staphylococcus pseudintermedius": {
        "Ampicillin":0.60,"Amoxicillin-Clavulanate":0.30,"Piperacillin-Tazobactam":0.25,
        "Cefazolin":0.30,"Cefuroxime":0.28,"Ceftriaxone":0.25,"Ceftazidime":0.35,"Cefepime":0.30,
        "Meropenem":0.05,"Imipenem":0.05,"Ertapenem":0.05,
        "Ciprofloxacin":0.35,"Levofloxacin":0.32,
        "Gentamicin":0.25,"Tobramycin":0.22,"Amikacin":0.12,
        "Vancomycin":0.01,"Teicoplanin":0.01,
        "Tetracycline":0.40,"Tigecycline":0.03,
        "Colistin":1.00,"Trimethoprim-Sulfamethoxazole":0.30,"Linezolid":0.02,
    },
    "Pseudomonas aeruginosa": {
        "Ampicillin":1.00,"Amoxicillin-Clavulanate":1.00,"Piperacillin-Tazobactam":0.25,
        "Cefazolin":1.00,"Cefuroxime":1.00,"Ceftriaxone":0.80,"Ceftazidime":0.25,"Cefepime":0.22,
        "Meropenem":0.20,"Imipenem":0.20,"Ertapenem":0.90,
        "Ciprofloxacin":0.22,"Levofloxacin":0.22,
        "Gentamicin":0.20,"Tobramycin":0.18,"Amikacin":0.12,
        "Vancomycin":1.00,"Teicoplanin":1.00,
        "Tetracycline":1.00,"Tigecycline":0.80,
        "Colistin":0.05,"Trimethoprim-Sulfamethoxazole":1.00,"Linezolid":1.00,
    },
    "Acinetobacter baumannii": {
        "Ampicillin":0.95,"Amoxicillin-Clavulanate":0.90,"Piperacillin-Tazobactam":0.75,
        "Cefazolin":0.95,"Cefuroxime":0.92,"Ceftriaxone":0.80,"Ceftazidime":0.70,"Cefepime":0.65,
        "Meropenem":0.55,"Imipenem":0.55,"Ertapenem":0.60,
        "Ciprofloxacin":0.65,"Levofloxacin":0.62,
        "Gentamicin":0.55,"Tobramycin":0.52,"Amikacin":0.40,
        "Vancomycin":1.00,"Teicoplanin":1.00,
        "Tetracycline":0.70,"Tigecycline":0.20,
        "Colistin":0.08,"Trimethoprim-Sulfamethoxazole":0.70,"Linezolid":1.00,
    },
    "Enterococcus faecium": {
        "Ampicillin":0.80,"Amoxicillin-Clavulanate":0.75,"Piperacillin-Tazobactam":0.70,
        "Cefazolin":1.00,"Cefuroxime":1.00,"Ceftriaxone":1.00,"Ceftazidime":1.00,"Cefepime":1.00,
        "Meropenem":0.70,"Imipenem":0.60,"Ertapenem":0.90,
        "Ciprofloxacin":0.50,"Levofloxacin":0.48,
        "Gentamicin":0.40,"Tobramycin":0.42,"Amikacin":0.38,
        "Vancomycin":0.25,"Teicoplanin":0.20,
        "Tetracycline":0.60,"Tigecycline":0.08,
        "Colistin":1.00,"Trimethoprim-Sulfamethoxazole":0.40,"Linezolid":0.05,
    },
    "Enterococcus faecalis": {
        "Ampicillin":0.05,"Amoxicillin-Clavulanate":0.10,"Piperacillin-Tazobactam":0.08,
        "Cefazolin":1.00,"Cefuroxime":1.00,"Ceftriaxone":0.90,"Ceftazidime":1.00,"Cefepime":0.90,
        "Meropenem":0.40,"Imipenem":0.30,"Ertapenem":0.80,
        "Ciprofloxacin":0.20,"Levofloxacin":0.18,
        "Gentamicin":0.30,"Tobramycin":0.35,"Amikacin":0.30,
        "Vancomycin":0.05,"Teicoplanin":0.05,
        "Tetracycline":0.45,"Tigecycline":0.05,
        "Colistin":1.00,"Trimethoprim-Sulfamethoxazole":0.30,"Linezolid":0.02,
    },
    "Mannheimia haemolytica": {
        "Ampicillin":0.30,"Amoxicillin-Clavulanate":0.15,"Piperacillin-Tazobactam":0.10,
        "Cefazolin":0.12,"Cefuroxime":0.10,"Ceftriaxone":0.08,"Ceftazidime":0.08,"Cefepime":0.07,
        "Meropenem":0.02,"Imipenem":0.02,"Ertapenem":0.02,
        "Ciprofloxacin":0.18,"Levofloxacin":0.15,
        "Gentamicin":0.20,"Tobramycin":0.18,"Amikacin":0.05,
        "Vancomycin":0.00,"Teicoplanin":0.00,
        "Tetracycline":0.50,"Tigecycline":0.04,
        "Colistin":0.00,"Trimethoprim-Sulfamethoxazole":0.28,"Linezolid":0.00,
    },
    "Pasteurella multocida": {
        "Ampicillin":0.15,"Amoxicillin-Clavulanate":0.08,"Piperacillin-Tazobactam":0.05,
        "Cefazolin":0.08,"Cefuroxime":0.06,"Ceftriaxone":0.05,"Ceftazidime":0.05,"Cefepime":0.04,
        "Meropenem":0.01,"Imipenem":0.01,"Ertapenem":0.01,
        "Ciprofloxacin":0.10,"Levofloxacin":0.08,
        "Gentamicin":0.12,"Tobramycin":0.10,"Amikacin":0.03,
        "Vancomycin":0.00,"Teicoplanin":0.00,
        "Tetracycline":0.35,"Tigecycline":0.03,
        "Colistin":0.00,"Trimethoprim-Sulfamethoxazole":0.22,"Linezolid":0.00,
    },
    "Enterobacter cloacae": {
        "Ampicillin":0.98,"Amoxicillin-Clavulanate":0.85,"Piperacillin-Tazobactam":0.30,
        "Cefazolin":0.90,"Cefuroxime":0.85,"Ceftriaxone":0.28,"Ceftazidime":0.22,"Cefepime":0.18,
        "Meropenem":0.08,"Imipenem":0.08,"Ertapenem":0.10,
        "Ciprofloxacin":0.25,"Levofloxacin":0.22,
        "Gentamicin":0.18,"Tobramycin":0.16,"Amikacin":0.06,
        "Vancomycin":0.00,"Teicoplanin":0.00,
        "Tetracycline":0.45,"Tigecycline":0.06,
        "Colistin":0.03,"Trimethoprim-Sulfamethoxazole":0.35,"Linezolid":0.00,
    },
    "Clostridioides difficile": {
        "Ampicillin":0.80,"Amoxicillin-Clavulanate":0.70,"Piperacillin-Tazobactam":0.60,
        "Cefazolin":0.90,"Cefuroxime":0.90,"Ceftriaxone":0.85,"Ceftazidime":0.88,"Cefepime":0.85,
        "Meropenem":0.10,"Imipenem":0.10,"Ertapenem":0.12,
        "Ciprofloxacin":0.45,"Levofloxacin":0.42,
        "Gentamicin":0.10,"Tobramycin":0.10,"Amikacin":0.05,
        "Vancomycin":0.02,"Teicoplanin":0.02,
        "Tetracycline":0.30,"Tigecycline":0.04,
        "Colistin":0.50,"Trimethoprim-Sulfamethoxazole":0.20,"Linezolid":0.05,
    },
}

# ─── Host-specific resistance modifiers ───────────────────────────────────────
HOST_RESISTANCE_BOOST = {
    "Human":   0.00,   # baseline (hospital setting included separately)
    "Bovine":  0.05,   # heavy antibiotic use in agriculture
    "Porcine": 0.10,   # highest agricultural antibiotic use
    "Poultry": 0.08,   # prophylactic use common
    "Canine":  0.03,
    "Feline":  0.02,
    "Equine":  0.02,
    "Wild Bird":  0.00,
    "Wild Boar":  0.03,
    "Water":   0.05,   # accumulates run-off resistance genes
    "Soil":    0.04,
    "Food":    0.07,   # end-product contamination
}

COUNTRIES     = ["Portugal","Spain","France","Germany","Italy","United Kingdom","Netherlands","Poland","Brazil","India","USA","South Africa"]
SEX_OPTIONS   = {"Human":["Male","Female"], "Bovine":["Male","Female","Unknown"],
                 "Porcine":["Male","Female","Unknown"], "Poultry":["Male","Female","Unknown"],
                 "Canine":["Male","Female"],"Feline":["Male","Female"],"Equine":["Male","Female"],
                 "Wild Bird":["Unknown"],"Wild Boar":["Male","Female","Unknown"],
                 "Water":["N/A"],"Soil":["N/A"],"Food":["N/A"]}

ZOONOTIC_RISK_SCORE = {"Very High": 4, "High": 3, "Moderate": 2, "Low": 1, "Negligible": 0}


def _classify_mdr(row: pd.Series) -> str:
    ab_cols  = [c for c in row.index if c in ALL_ANTIBIOTICS]
    resistant = row[ab_cols].sum()
    total     = len(ab_cols)
    pct       = resistant / total
    if resistant >= 3 and pct >= 0.5:
        return "MDR"
    elif resistant >= 1:
        return "Intermediate"
    return "Susceptible"


def generate_dataset(n: int = 3000, save_path: str | None = None) -> pd.DataFrame:
    """
    Generate a One Health synthetic AMR dataset.

    Parameters
    ----------
    n : int
        Total number of isolate records.
    save_path : str | None
        Path to save CSV.
    """
    host_list    = list(HOST_WEIGHTS.keys())
    host_weights = list(HOST_WEIGHTS.values())

    chosen_hosts = RNG.choice(host_list, size=n, p=host_weights)

    records = []
    for i, host in enumerate(chosen_hosts):
        host_info = HOST_CATEGORIES[host]

        # Choose a compatible bacterial species
        compatible = [sp for sp, info in BACTERIA.items() if host in info["hosts"]]
        if not compatible:
            compatible = ["Escherichia coli"]
        species = RNG.choice(compatible)

        bact_info = BACTERIA[species]
        r_probs   = RESISTANCE_PROBS_BASE[species]
        host_boost = HOST_RESISTANCE_BOOST[host]

        # Setting / sampling context
        setting = RNG.choice(host_info["setting"])
        country = RNG.choice(COUNTRIES)
        sex_opts = SEX_OPTIONS.get(host, ["Unknown"])
        sex      = RNG.choice(sex_opts)
        year     = int(RNG.choice(range(2015, 2025)))

        # Age: meaningful only for animals/humans
        if host == "Human":
            age = int(np.clip(RNG.normal(55, 22), 0, 98))
        elif host in ["Bovine","Porcine","Poultry","Equine"]:
            age = int(np.clip(RNG.exponential(3), 0, 20))
        elif host in ["Canine","Feline"]:
            age = int(np.clip(RNG.normal(6, 4), 0, 18))
        elif host in ["Wild Bird","Wild Boar"]:
            age = int(np.clip(RNG.exponential(2), 0, 10))
        else:
            age = -1  # N/A for environment

        # Antibiotic exposure prior (animals more likely in farming)
        prior_ab = int(RNG.random() < (0.6 if host in ["Bovine","Porcine","Poultry"] else
                                       0.4 if host == "Human" else 0.2))

        icu_or_intensive = int(RNG.random() < (0.15 if host == "Human" else 0.05))

        esbl_rate = bact_info["esbl_rates"].get(host, 0.0)
        esbl      = int(RNG.random() < esbl_rate)

        mrsa = int(species == "Staphylococcus aureus" and
                   host in ["Human","Porcine"] and RNG.random() < 0.25)

        vre = int(species == "Enterococcus faecium" and RNG.random() < 0.18)

        carbapenemase = int(
            species in ["Klebsiella pneumoniae","Acinetobacter baumannii"] and
            RNG.random() < (0.08 if host == "Human" else 0.02)
        )

        # Antibiogram
        abx_profile = {}
        for ab, base_prob in r_probs.items():
            adjusted = min(base_prob + host_boost
                           + (0.10 if prior_ab else 0.0)
                           + (0.08 if icu_or_intensive else 0.0), 0.99)
            abx_profile[ab] = int(RNG.random() < adjusted)

        if esbl:
            for ab in ["Ceftriaxone","Ceftazidime","Cefepime","Cefuroxime","Cefazolin"]:
                if ab in abx_profile:
                    abx_profile[ab] = 1
        if mrsa:
            abx_profile["Ampicillin"] = 1
            abx_profile["Amoxicillin-Clavulanate"] = 1
        if vre:
            abx_profile["Vancomycin"] = 1
            abx_profile["Teicoplanin"] = 1

        record = {
            "isolate_id":     f"ISO-{i+1:05d}",
            "host_species":   host,
            "host_type":      host_info["type"],
            "host_icon":      host_info["icon"],
            "species":        species,
            "gram_stain":     bact_info["gram"],
            "family":         bact_info["family"],
            "zoonotic":       int(bact_info["zoonotic"]),
            "zoonotic_risk":  bact_info["zoonotic_risk"],
            "zoonotic_score": ZOONOTIC_RISK_SCORE[bact_info["zoonotic_risk"]],
            "setting":        setting,
            "country":        country,
            "host_sex":       sex,
            "host_age":       age,
            "prior_antibiotic_exposure": prior_ab,
            "intensive_care_or_farming": icu_or_intensive,
            "esbl":           esbl,
            "carbapenemase":  carbapenemase,
            "mrsa":           mrsa,
            "vre":            vre,
            "year":           year,
            **abx_profile,
        }
        records.append(record)

    df = pd.DataFrame(records)
    df["total_resistant"] = df[ALL_ANTIBIOTICS].sum(axis=1)
    df["resistance_rate"] = (df["total_resistant"] / len(ALL_ANTIBIOTICS)).round(3)
    df["mdr_phenotype"]   = df.apply(_classify_mdr, axis=1)
    df["is_mdr"]          = (df["mdr_phenotype"] == "MDR").astype(int)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"[✓] One Health dataset saved → {save_path}  ({len(df):,} records)")

    return df


# ── Convenience exports ───────────────────────────────────────────────────────
BACTERIA_NAMES   = list(BACTERIA.keys())
HOST_NAMES       = list(HOST_CATEGORIES.keys())
SPECIMEN_TYPES   = ["Urine","Blood","Respiratory","Wound","CSF","Stool",
                     "Nasal swab","Ear swab","Wound swab","Milk","Faeces",
                     "Surface swab","Water sample","Soil sample","Feed"]
WARDS            = ["ICU","Emergency","Surgical","Medical","Paediatrics",
                     "Oncology","Veterinary ICU","Farm","Environment"]
HOSPITALS        = ["Hospital A (Tertiary)","Hospital B (Secondary)",
                     "Hospital C (Primary)","Hospital D (University)",
                     "Veterinary Hospital","Farm Clinic","Reference Lab"]

if __name__ == "__main__":
    df = generate_dataset(n=3000, save_path="../data/amr_synthetic_dataset.csv")
    print(df.head())
    print(f"\nMDR rate: {df['is_mdr'].mean():.1%}")
    print(df["host_species"].value_counts())
    print(df["species"].value_counts())
