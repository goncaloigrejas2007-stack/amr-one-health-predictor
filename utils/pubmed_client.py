"""
pubmed_client.py — Live PubMed & NCBI Entrez Evidence Client
=============================================================
Fetches real peer-reviewed scientific literature and molecular resistance
evidence directly from the National Center for Biotechnology Information (NCBI)
PubMed database via the official Entrez E-Utilities API.

Author: Gonçalo Igrejas
"""

import json
import logging
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Base NCBI E-Utilities endpoints
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

# Molecular resistance mechanisms mapping (Pathogen / Class -> Genes & Mechanisms)
RESISTANCE_MECHANISMS_DB = {
    "Klebsiella pneumoniae": {
        "Carbapenems": ["blaKPC-2", "blaKPC-3", "blaNDM-1", "blaOXA-48"],
        "Polymyxins": ["mgrB inactivation", "pmrA/pmrB mutations", "mcr-1"],
        "Cephalosporins": ["blaCTX-M-15 (ESBL)", "blaSHV-12", "AmpC"],
    },
    "Escherichia coli": {
        "Beta-lactams": ["blaTEM-1", "blaCTX-M-14", "blaCTX-M-15"],
        "Fluoroquinolones": ["gyrA (S83L, D87N)", "parC (S80I)", "qnrS1"],
        "Polymyxins": ["mcr-1 plasmid-mediated colistin resistance"],
    },
    "Staphylococcus aureus": {
        "Beta-lactams": ["mecA (PBP2a)", "mecC", "blaZ penicillinase"],
        "Glycopeptides": ["vanA operon (VRSA)", "cell wall thickening (VISA)"],
        "Oxazolidinones": ["cfr 23S rRNA methyltransferase", "linezolid G2576T"],
    },
    "Pseudomonas aeruginosa": {
        "Carbapenems": ["OprD porin loss", "blaVIM", "blaIMP metallo-beta-lactamases"],
        "Aminoglycosides": ["MexXY-OprM efflux pump", "aac(6')-Ib aminoglycoside acetyltransferase"],
        "Cephalosporins": ["Overexpressed AmpC cephalosporinase"],
    },
    "Acinetobacter baumannii": {
        "Carbapenems": ["blaOXA-23", "blaOXA-51-like", "blaOXA-58", "blaNDM"],
        "Polymyxins": ["lpxA/C/D lipid A mutations", "pmrA/B upregulation"],
    },
    "Enterococcus faecium": {
        "Glycopeptides": ["vanA (high-level vancomycin & teicoplanin)", "vanB operon"],
        "Beta-lactams": ["pbp5 mutations (low affinity PBP5)"],
        "Oxazolidinones": ["optrA", "poxtA", "23S rRNA mutations"],
    },
    "Salmonella enterica": {
        "Fluoroquinolones": ["gyrA/B, parC mutations", "qnrB, qnrS"],
        "Cephalosporins": ["blaCTX-M-1", "blaCMY-2 (AmpC)", "blaTEM-52"],
        "Polymyxins": ["mcr-1, mcr-3, mcr-9 in livestock and food isolates"],
    },
    "Campylobacter jejuni": {
        "Fluoroquinolones": ["gyrA (T86I high-level ciprofloxacin resistance)"],
        "Macrolides": ["23S rRNA A2075G mutation"],
        "Tetracyclines": ["tet(O) ribosomal protection protein"],
    },
}

# Curated benchmark papers for offline fallback
FALLBACK_PAPERS = [
    {
        "pmid": "35065702",
        "title": "Global burden of bacterial antimicrobial resistance in 2019: a systematic analysis.",
        "authors": "Murray CJ, Ikuta KS, Sharara F, Swetschinski L, et al.",
        "source": "The Lancet",
        "pubdate": "2022 Feb 12",
        "doi": "10.1016/S0140-6736(21)02724-0",
        "url": "https://pubmed.ncbi.nlm.nih.gov/35065702/",
    },
    {
        "pmid": "22452372",
        "title": "Multidrug-resistant, extensively drug-resistant and pandrug-resistant bacteria: an international expert proposal for interim standard definitions for acquired resistance.",
        "authors": "Magiorakos AP, Srinivasan A, Carey RB, Carmeli Y, et al.",
        "source": "Clin Microbiol Infect",
        "pubdate": "2012 Mar",
        "doi": "10.1111/j.1469-0691.2011.03570.x",
        "url": "https://pubmed.ncbi.nlm.nih.gov/22452372/",
    },
    {
        "pmid": "26603172",
        "title": "Emergence of plasmid-mediated colistin resistance mechanism MCR-1 in animals and human beings in China: a microbiological and molecular biological study.",
        "authors": "Liu YY, Wang Y, Walsh TR, Yi LX, et al.",
        "source": "Lancet Infect Dis",
        "pubdate": "2016 Feb",
        "doi": "10.1016/S1473-3099(15)00424-7",
        "url": "https://pubmed.ncbi.nlm.nih.gov/26603172/",
    },
]


def search_pubmed(query: str, max_results: int = 5, timeout: int = 8) -> List[Dict]:
    """
    Search PubMed via NCBI Entrez E-Utilities.
    Returns a list of structured article dictionaries with title, authors, source, pubdate, pmid, and URL.
    """
    try:
        # Step 1: ESearch to retrieve matching PMIDs
        params = {
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": str(max_results),
            "sort": "pub_date",
        }
        esearch_url = f"{ESEARCH_URL}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(esearch_url, headers={"User-Agent": "AMR-Predictor-OneHealth/1.0"})

        with urllib.request.urlopen(req, timeout=timeout) as response:
            search_data = json.loads(response.read().decode("utf-8"))

        id_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []

        # Step 2: ESummary to get article metadata
        summary_params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json",
        }
        summary_url = f"{ESUMMARY_URL}?{urllib.parse.urlencode(summary_params)}"
        req_sum = urllib.request.Request(summary_url, headers={"User-Agent": "AMR-Predictor-OneHealth/1.0"})

        with urllib.request.urlopen(req_sum, timeout=timeout) as sum_response:
            summary_data = json.loads(sum_response.read().decode("utf-8"))

        results = []
        result_dict = summary_data.get("result", {})

        for pmid in id_list:
            item = result_dict.get(pmid)
            if not item:
                continue

            authors_list = item.get("authors", [])
            author_names = [a.get("name", "") for a in authors_list if "name" in a]
            if len(author_names) > 3:
                authors_str = f"{', '.join(author_names[:3])} et al."
            elif author_names:
                authors_str = ", ".join(author_names)
            else:
                authors_str = "Unknown Authors"

            # Extract DOI if present
            article_ids = item.get("articleids", [])
            doi = ""
            for aid in article_ids:
                if aid.get("idtype") == "doi":
                    doi = aid.get("value", "")
                    break

            results.append({
                "pmid": pmid,
                "title": item.get("title", "No Title Available").rstrip("."),
                "authors": authors_str,
                "source": item.get("source", "Journal"),
                "pubdate": item.get("pubdate", ""),
                "doi": doi,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })

        return results

    except Exception as e:
        logger.warning(f"PubMed API search error for '{query}': {e}")
        # Return fallback papers if search fails
        return FALLBACK_PAPERS[:max_results]


def get_amr_evidence(
    bacteria: Optional[str] = None,
    antibiotic: Optional[str] = None,
    host: Optional[str] = None,
    max_results: int = 5,
) -> List[Dict]:
    """
    Generates an optimized PubMed query for AMR evidence tailored to
    bacteria, antibiotic, and host context.
    """
    query_parts = []

    if bacteria:
        query_parts.append(f'"{bacteria}"[Title/Abstract]')

    if antibiotic:
        query_parts.append(f'("{antibiotic}"[Title/Abstract] AND resistance[Title/Abstract])')
    else:
        query_parts.append('("antimicrobial resistance"[Title/Abstract] OR "multidrug-resistant"[Title/Abstract])')

    if host:
        if host in ["Human", "Clinical"]:
            query_parts.append('(human OR clinical OR patient)')
        elif host in ["Bovine", "Porcine", "Poultry", "Livestock"]:
            query_parts.append(f'("{host}" OR livestock OR veterinary OR zoonotic)')
        elif host in ["Canine", "Feline", "Equine", "Companion"]:
            query_parts.append(f'("{host}" OR "companion animal" OR veterinary)')
        elif host in ["Water", "Soil", "Environment", "Food"]:
            query_parts.append(f'("{host}" OR environmental OR wastewater OR "One Health")')
        elif host in ["Wild Bird", "Wild Boar", "Wildlife"]:
            query_parts.append(f'("{host}" OR wildlife OR reservoir)')

    query = " AND ".join(query_parts)
    return search_pubmed(query, max_results=max_results)


def get_known_molecular_mechanisms(bacteria: str) -> Dict[str, List[str]]:
    """
    Returns curated known resistance genes/mechanisms for a given bacterial species.
    """
    return RESISTANCE_MECHANISMS_DB.get(bacteria, {
        "Multi-drug resistance": ["Efflux pumps (RND / ABC families)", "Porin alterations", "Target mutations"]
    })
