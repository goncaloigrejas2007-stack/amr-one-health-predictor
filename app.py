"""
app.py — AMR-Predictor One Health Dashboard
=============================================
Interactive Streamlit app for One Health Antimicrobial Resistance analysis.
Covers humans, livestock, companion animals, wildlife and environmental samples.

Author : Gonçalo Igrejas
"""

import sys, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.data_generator import (
    generate_dataset, ALL_ANTIBIOTICS, ANTIBIOTIC_CLASSES,
    BACTERIA, HOST_CATEGORIES, HOST_NAMES, COUNTRIES,
    ZOONOTIC_RISK_SCORE,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AMR One Health · Clinical & Veterinary Microbiology AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Register Plotly theme ─────────────────────────────────────────────────────
pio.templates["amr_dark"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#060d1f",
        plot_bgcolor="#0d1b2e",
        font=dict(family="Inter", color="#e2e8f0"),
        colorway=["#3b82f6","#14b8a6","#8b5cf6","#f59e0b","#f43f5e",
                  "#06b6d4","#84cc16","#ec4899","#fb923c","#a78bfa"],
        xaxis=dict(gridcolor="#1a2d45", zerolinecolor="#1a2d45"),
        yaxis=dict(gridcolor="#1a2d45", zerolinecolor="#1a2d45"),
    )
)
AMR_TEMPLATE = "amr_dark"

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.block-container { padding: 1.5rem 2rem 2rem 2rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#030b1a 0%,#060d1f 100%);
    border-right: 1px solid rgba(59,130,246,0.12);
}

/* ── KPI cards ── */
.metric-card {
    background: linear-gradient(135deg,#0d1b2e 0%,#132035 100%);
    border: 1px solid rgba(59,130,246,0.18);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    transition: transform .2s, box-shadow .2s;
}
.metric-card:hover { transform:translateY(-3px); box-shadow:0 10px 28px rgba(59,130,246,.18); }
.metric-value {
    font-size:1.9rem; font-weight:800;
    background:linear-gradient(90deg,#3b82f6,#14b8a6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.metric-label { font-size:.75rem; color:#64748b; text-transform:uppercase; letter-spacing:.08em; margin-top:3px; }

/* ── Section headers ── */
.section-header {
    font-size:1rem; font-weight:700; color:#f1f5f9;
    border-left:3px solid #3b82f6; padding-left:.7rem;
    margin-bottom:.9rem; letter-spacing:.01em;
}

/* ── Badges ── */
.badge { display:inline-block; padding:3px 11px; border-radius:20px; font-size:.73rem; font-weight:600; }
.badge-mdr   { background:#f43f5e22; color:#f43f5e; border:1px solid #f43f5e44; }
.badge-sus   { background:#14b8a622; color:#14b8a6; border:1px solid #14b8a644; }
.badge-high  { background:#f59e0b22; color:#f59e0b; border:1px solid #f59e0b44; }
.badge-host  { background:#8b5cf622; color:#8b5cf6; border:1px solid #8b5cf644; }

/* ── Pred box ── */
.pred-box { border-radius:16px; padding:1.5rem; text-align:center; font-size:1.35rem; font-weight:800; margin:.8rem 0; }
.pred-mdr { background:linear-gradient(135deg,#f43f5e22,#f43f5e11); border:2px solid #f43f5e55; color:#f43f5e; }
.pred-sus { background:linear-gradient(135deg,#14b8a622,#14b8a611); border:2px solid #14b8a655; color:#14b8a6; }

/* ── One Health pill ── */
.oh-pill {
    display:inline-flex; align-items:center; gap:6px;
    background:#8b5cf611; border:1px solid #8b5cf633;
    border-radius:24px; padding:4px 14px;
    font-size:.78rem; font-weight:600; color:#a78bfa;
    margin:3px;
}

/* ── Zoonotic risk bar ── */
.risk-bar { display:flex; align-items:center; gap:8px; margin:4px 0; }
.risk-fill { height:8px; border-radius:4px; }

[data-testid="metric-container"] {
    background:#0d1b2e; border:1px solid rgba(59,130,246,.15); border-radius:10px; padding:1rem;
}

::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#060d1f; }
::-webkit-scrollbar-thumb { background:#1e3a5f; border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS & DATA
# ═══════════════════════════════════════════════════════════════════════════════

HOST_TYPE_COLORS = {
    "Human":       "#3b82f6",
    "Livestock":   "#f59e0b",
    "Companion":   "#14b8a6",
    "Wildlife":    "#84cc16",
    "Environment": "#8b5cf6",
}

HOST_TYPE_RGBA = {
    "Human":       "rgba(59, 130, 246, 0.15)",
    "Livestock":   "rgba(245, 158, 11, 0.15)",
    "Companion":   "rgba(20, 184, 166, 0.15)",
    "Wildlife":    "rgba(132, 204, 22, 0.15)",
    "Environment": "rgba(139, 92, 246, 0.15)",
}

ZOONOTIC_COLORS = {"Very High":"#f43f5e","High":"#f97316","Moderate":"#f59e0b",
                   "Low":"#14b8a6","Negligible":"#64748b"}


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    p = ROOT / "data" / "amr_synthetic_dataset.csv"
    if p.exists():
        return pd.read_csv(p)
    return generate_dataset(n=3000, save_path=str(p))


@st.cache_resource(show_spinner=False)
def get_model():
    mp  = ROOT / "models" / "amr_rf_model.joblib"
    fp  = ROOT / "models" / "feature_names.joblib"
    mep = ROOT / "models" / "model_metrics.joblib"
    if mp.exists():
        return joblib.load(mp), joblib.load(fp), joblib.load(mep)
    return None, None, None


def metric_card(label, value, sub=""):
    return (f'<div class="metric-card"><div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div>'
            + (f'<div style="font-size:.72rem;color:#475569;margin-top:2px">{sub}</div>' if sub else "")
            + "</div>")


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 .4rem">
        <div style="font-size:2.6rem">🌍</div>
        <div style="font-size:1.05rem;font-weight:800;color:#f1f5f9">AMR One Health</div>
        <div style="font-size:.7rem;color:#475569;margin-top:2px">
            Human · Animal · Environment
        </div>
    </div>
    <hr style="border-color:#0f2035;margin:.7rem 0">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊 Dashboard","🌍 One Health","🔬 Exploratory Analysis","🤖 ML Model","🧬 AMR Predictor"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#0f2035'>", unsafe_allow_html=True)
    st.markdown("**Filters**")

    df_full = get_dataset()

    sel_host_types = st.multiselect(
        "Host Category",
        sorted(df_full["host_type"].unique()),
        default=sorted(df_full["host_type"].unique()),
    )
    sel_hosts = st.multiselect(
        "Host Species",
        sorted(df_full["host_species"].unique()),
        default=sorted(df_full["host_species"].unique()),
    )
    sel_year = st.slider(
        "Year",
        int(df_full["year"].min()), int(df_full["year"].max()),
        (int(df_full["year"].min()), int(df_full["year"].max())),
    )

    st.markdown("<hr style='border-color:#0f2035'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:.7rem;color:#334155;line-height:1.7">
        <b>n =</b> 3,000 synthetic isolates<br>
        <b>Hosts:</b> 12 species categories<br>
        <b>Bacteria:</b> 13 species<br>
        <b>Antibiotics:</b> 22 agents, 10 classes<br>
        <b>Basis:</b> ECDC/EFSA/WHO 2015-2024<br><br>
        <i>⚠️ Synthetic data — educational only.</i>
    </div>
    """, unsafe_allow_html=True)

df = df_full[
    df_full["host_type"].isin(sel_host_types) &
    df_full["host_species"].isin(sel_hosts) &
    df_full["year"].between(sel_year[0], sel_year[1])
].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📊 Dashboard":
    st.markdown("""
    <h1 style="margin:0;font-size:1.75rem;font-weight:800;color:#f1f5f9">
        AMR One Health — Intelligence Platform
    </h1>
    <p style="color:#475569;font-size:.88rem;margin-top:4px">
        Antimicrobial Resistance across Humans · Livestock · Companion Animals · Wildlife · Environment
    </p>
    <hr style="border-color:#0f2035;margin:1rem 0">
    """, unsafe_allow_html=True)

    # ── KPIs ────────────────────────────────────────────────────────────────
    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: st.markdown(metric_card("Total Isolates", f"{len(df):,}", "filtered"), unsafe_allow_html=True)
    with k2: st.markdown(metric_card("MDR Rate", f"{df['is_mdr'].mean():.1%}", "multi-drug resistant"), unsafe_allow_html=True)
    with k3: st.markdown(metric_card("Zoonotic Isolates", f"{df['zoonotic'].mean():.1%}", "potential human risk"), unsafe_allow_html=True)
    with k4: st.markdown(metric_card("ESBL Producers", f"{df['esbl'].mean():.1%}", "extended-spectrum β-lactamase"), unsafe_allow_html=True)
    with k5: st.markdown(metric_card("Host Categories", f"{df['host_type'].nunique()}", "One Health domains"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: Host type donut + MDR by host type ────────────────────────────
    c1,c2 = st.columns([1,1.3])
    with c1:
        st.markdown('<div class="section-header">Isolates by Host Category</div>', unsafe_allow_html=True)
        ht = df["host_type"].value_counts()
        fig = go.Figure(go.Pie(
            labels=ht.index, values=ht.values, hole=0.55,
            textinfo="label+percent", textfont_size=11,
            marker=dict(colors=[HOST_TYPE_COLORS.get(h,"#94a3b8") for h in ht.index],
                        line=dict(color="#060d1f",width=2)),
        ))
        fig.update_layout(template=AMR_TEMPLATE, height=300, showlegend=False,
                          margin=dict(t=10,b=10,l=10,r=10),
                          annotations=[dict(text="Host",x=.5,y=.5,
                                            font_size=13,font_color="#94a3b8",showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-header">MDR Rate by Host Species</div>', unsafe_allow_html=True)
        mdr_host = (df.groupby(["host_species","host_type"])["is_mdr"]
                    .mean().reset_index().sort_values("is_mdr",ascending=True))
        mdr_host["color"] = mdr_host["host_type"].map(HOST_TYPE_COLORS)
        mdr_host["label"] = mdr_host["is_mdr"].map(lambda x: f"{x:.1%}")
        fig = px.bar(mdr_host, x="is_mdr", y="host_species", orientation="h",
                     color="host_type", text="label",
                     color_discrete_map=HOST_TYPE_COLORS,
                     labels={"is_mdr":"MDR Rate","host_species":"","host_type":"Category"})
        fig.update_traces(textposition="outside", textfont_size=10)
        fig.update_layout(template=AMR_TEMPLATE, height=300,
                          margin=dict(t=10,b=10,l=10,r=80),
                          legend=dict(orientation="h",y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Resistance heatmap ────────────────────────────────────────────
    st.markdown('<div class="section-header">Resistance Rate Heatmap — Bacterial Species × Antibiotic</div>', unsafe_allow_html=True)
    heat = df.groupby("species")[ALL_ANTIBIOTICS].mean().round(2)
    fig = go.Figure(go.Heatmap(
        z=heat.values, x=heat.columns.tolist(), y=heat.index.tolist(),
        colorscale="RdYlGn_r", zmid=0.5,
        text=np.round(heat.values,2), texttemplate="%{text}",
        textfont=dict(size=9),
        colorbar=dict(title="Rate",tickformat=".0%",len=0.8),
    ))
    fig.update_layout(template=AMR_TEMPLATE, height=360,
                      margin=dict(t=10,b=90,l=240,r=10),
                      xaxis=dict(tickangle=-45,tickfont_size=10),
                      yaxis=dict(tickfont_size=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: Temporal trend + ESBL by host ────────────────────────────────
    c3,c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-header">MDR Trend by Host Category (2015–2024)</div>', unsafe_allow_html=True)
        trend = df.groupby(["year","host_type"])["is_mdr"].mean().reset_index()
        fig = px.line(trend, x="year", y="is_mdr", color="host_type",
                      color_discrete_map=HOST_TYPE_COLORS, markers=True,
                      labels={"is_mdr":"MDR Rate","year":"Year","host_type":"Host Category"})
        fig.update_layout(template=AMR_TEMPLATE, height=300,
                          yaxis_tickformat=".0%",
                          legend=dict(orientation="h",y=-0.25))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown('<div class="section-header">MDR Rate by Country</div>', unsafe_allow_html=True)
        country_mdr = df.groupby("country")["is_mdr"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(country_mdr, x="country", y="is_mdr",
                     color="is_mdr", color_continuous_scale="Reds",
                     text=country_mdr["is_mdr"].map(lambda x: f"{x:.1%}"),
                     labels={"is_mdr":"MDR Rate","country":""})
        fig.update_traces(textposition="outside")
        fig.update_layout(template=AMR_TEMPLATE, height=300,
                          coloraxis_showscale=False,
                          xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ONE HEALTH
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🌍 One Health":
    st.markdown("""
    <h1 style="margin:0;font-size:1.75rem;font-weight:800;color:#f1f5f9">
        One Health AMR Analysis
    </h1>
    <p style="color:#475569;font-size:.88rem;margin-top:4px">
        Zoonotic transmission networks · Cross-species resistance comparison · Shared bacterial threats
    </p>
    <hr style="border-color:#0f2035;margin:1rem 0">
    """, unsafe_allow_html=True)

    # ── One Health concept banner ────────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1b2e,#132035);border:1px solid rgba(139,92,246,.25);
                border-radius:14px;padding:1.1rem 1.4rem;margin-bottom:1.2rem">
        <div style="font-size:.85rem;color:#c4b5fd;font-weight:600;margin-bottom:.4rem">
            🌐 What is One Health?
        </div>
        <div style="font-size:.8rem;color:#94a3b8;line-height:1.7">
            <b>One Health</b> is a WHO/FAO/OIE collaborative framework recognising that human health,
            animal health and ecosystem health are deeply interconnected. AMR spreads across these
            boundaries through direct contact, food chains, water systems and shared environments.
            ~75% of emerging infectious diseases are zoonotic — originating in animals.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔗 Zoonotic Network","📊 Cross-Species Comparison","🦠 Shared Pathogens","🌱 Environmental AMR"]
    )

    # ── Tab 1: Zoonotic Sankey ────────────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">AMR Transmission Network — Sankey Diagram</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:.78rem;color:#64748b;margin-bottom:.8rem">
        Flow represents shared zoonotic bacterial isolates between host categories.
        Thickness = number of isolates with High/Very High zoonotic risk.
        </div>""", unsafe_allow_html=True)

        # Build Sankey: source=host_type → target=species (zoonotic only)
        zon = df[df["zoonotic"] == 1].copy()
        zon_grp = (zon.groupby(["host_type","species"]).size().reset_index(name="count"))
        zon_grp = zon_grp[zon_grp["count"] >= 5]

        host_types = sorted(zon_grp["host_type"].unique().tolist())
        species_list = sorted(zon_grp["species"].unique().tolist())
        nodes = host_types + species_list
        node_idx = {n: i for i, n in enumerate(nodes)}

        colors_nodes = (
            [HOST_TYPE_COLORS.get(h,"#94a3b8") for h in host_types] +
            ["rgba(99,102,241,0.8)"] * len(species_list)
        )

        sources = [node_idx[r["host_type"]] for _, r in zon_grp.iterrows()]
        targets = [node_idx[r["species"]]   for _, r in zon_grp.iterrows()]
        values  = zon_grp["count"].tolist()

        fig = go.Figure(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=15, thickness=22, line=dict(color="#060d1f",width=.5),
                label=nodes, color=colors_nodes,
                hovertemplate="<b>%{label}</b><br>Flow: %{value}<extra></extra>",
            ),
            link=dict(
                source=sources, target=targets, value=values,
                color=["rgba(59,130,246,0.20)"] * len(sources),
                hovertemplate="<b>%{source.label}</b> → <b>%{target.label}</b><br>Isolates: %{value}<extra></extra>",
            ),
        ))
        fig.update_layout(template=AMR_TEMPLATE, height=480, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)

        # Zoonotic risk breakdown
        st.markdown('<div class="section-header">Zoonotic Risk Profile by Bacterial Species</div>', unsafe_allow_html=True)
        zr = df.groupby(["species","zoonotic_risk"]).size().reset_index(name="count")
        fig2 = px.bar(zr, x="species", y="count", color="zoonotic_risk",
                      color_discrete_map=ZOONOTIC_COLORS,
                      category_orders={"zoonotic_risk":["Very High","High","Moderate","Low","Negligible"]},
                      labels={"count":"Isolates","species":"","zoonotic_risk":"Zoonotic Risk"})
        fig2.update_layout(template=AMR_TEMPLATE, height=320,
                           xaxis_tickangle=-30,
                           legend=dict(orientation="h",y=-0.3))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: Cross-species comparison ──────────────────────────────────────
    with tab2:
        col_a, col_b = st.columns(2)
        with col_a:
            ab_sel = st.selectbox("Select antibiotic", ALL_ANTIBIOTICS, index=ALL_ANTIBIOTICS.index("Ciprofloxacin"))
        with col_b:
            bact_sel = st.selectbox("Select bacteria", sorted(df["species"].unique()))

        st.markdown(f'<div class="section-header">Resistance to {ab_sel} across Host Species</div>', unsafe_allow_html=True)
        cross = df.groupby("host_species")[ab_sel].mean().sort_values(ascending=False).reset_index()
        cross.columns = ["host_species","resistance"]
        cross["host_type"] = cross["host_species"].map(
            lambda h: HOST_CATEGORIES.get(h,{}).get("type","Other"))
        cross["icon"] = cross["host_species"].map(
            lambda h: HOST_CATEGORIES.get(h,{}).get("icon",""))
        cross["label"] = cross["resistance"].map(lambda x: f"{x:.1%}")

        fig3 = px.bar(cross, x="host_species", y="resistance",
                      color="host_type", text="label",
                      color_discrete_map=HOST_TYPE_COLORS,
                      labels={"resistance":f"Resistance to {ab_sel}","host_species":"Host"})
        fig3.add_hline(y=0.5, line_dash="dash", line_color="#f43f5e",
                       annotation_text="50% threshold")
        fig3.update_traces(textposition="outside")
        fig3.update_layout(template=AMR_TEMPLATE, height=350,
                           yaxis_tickformat=".0%",
                           legend=dict(orientation="h",y=-0.2))
        st.plotly_chart(fig3, use_container_width=True)

        # Radar chart per host type for selected bacteria
        st.markdown(f'<div class="section-header">Resistance Profile of {bact_sel} across Host Types (Radar)</div>', unsafe_allow_html=True)
        top_abs = ["Ampicillin","Ciprofloxacin","Ceftriaxone","Meropenem",
                   "Gentamicin","Tetracycline","Trimethoprim-Sulfamethoxazole","Colistin"]
        bact_df = df[df["species"] == bact_sel]
        radar_data = bact_df.groupby("host_type")[top_abs].mean()

        fig4 = go.Figure()
        for ht_name, row in radar_data.iterrows():
            fig4.add_trace(go.Scatterpolar(
                r=row.tolist() + [row.tolist()[0]],
                theta=top_abs + [top_abs[0]],
                fill="toself", name=ht_name,
                line=dict(color=HOST_TYPE_COLORS.get(ht_name, "#94a3b8"), width=2),
                fillcolor=HOST_TYPE_RGBA.get(ht_name, "rgba(148, 163, 184, 0.15)"),
            ))
        fig4.update_layout(template=AMR_TEMPLATE, height=420,
                           polar=dict(
                               bgcolor="#0d1b2e",
                               radialaxis=dict(visible=True, range=[0,1],
                                               tickformat=".0%", gridcolor="#1a2d45"),
                               angularaxis=dict(gridcolor="#1a2d45"),
                           ),
                           showlegend=True, legend=dict(orientation="h",y=-0.1))
        st.plotly_chart(fig4, use_container_width=True)

    # ── Tab 3: Shared pathogens ───────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Bacteria Shared Across Host Categories</div>', unsafe_allow_html=True)

        # Bubble chart: species × host_type, size=count, color=MDR rate
        bubble = (df.groupby(["species","host_type"])
                  .agg(count=("is_mdr","count"), mdr_rate=("is_mdr","mean"))
                  .reset_index())
        bubble = bubble[bubble["count"] >= 5]

        fig5 = px.scatter(
            bubble, x="host_type", y="species",
            size="count", color="mdr_rate",
            color_continuous_scale="RdYlGn_r",
            range_color=[0,1],
            size_max=45,
            labels={"host_type":"Host Category","species":"Bacterial Species",
                    "mdr_rate":"MDR Rate","count":"Isolates"},
            text=bubble["mdr_rate"].map(lambda x: f"{x:.0%}"),
        )
        fig5.update_traces(textfont_size=9)
        fig5.update_layout(template=AMR_TEMPLATE, height=500,
                           coloraxis_colorbar=dict(title="MDR Rate",tickformat=".0%"))
        st.plotly_chart(fig5, use_container_width=True)

        # Table of zoonotic species
        st.markdown('<div class="section-header">Zoonotic Pathogen Reference</div>', unsafe_allow_html=True)
        bact_tbl = []
        for sp, info in BACTERIA.items():
            if info["zoonotic"]:
                bact_tbl.append({
                    "Species": sp,
                    "Gram": info["gram"],
                    "Hosts": ", ".join(info["hosts"]),
                    "Zoonotic Risk": info["zoonotic_risk"],
                    "MDR Rate (dataset)": f'{df[df["species"]==sp]["is_mdr"].mean():.1%}' if sp in df["species"].values else "—",
                })
        tbl_df = pd.DataFrame(bact_tbl).sort_values("Zoonotic Risk",
            key=lambda s: s.map({"Very High":0,"High":1,"Moderate":2,"Low":3,"Negligible":4}))
        st.dataframe(tbl_df, use_container_width=True, hide_index=True)

    # ── Tab 4: Environmental AMR ──────────────────────────────────────────────
    with tab4:
        env = df[df["host_type"] == "Environment"].copy()
        if len(env) < 5:
            st.info("No environmental samples in current filter. Expand host filters.")
        else:
            st.markdown('<div class="section-header">Environmental AMR by Sample Type</div>', unsafe_allow_html=True)
            env_res = env.groupby("host_species")[ALL_ANTIBIOTICS].mean()
            fig6 = go.Figure(go.Heatmap(
                z=env_res.values, x=env_res.columns.tolist(),
                y=env_res.index.tolist(),
                colorscale="Purples", zmid=0.4,
                text=np.round(env_res.values,2), texttemplate="%{text}",
                textfont=dict(size=10),
            ))
            fig6.update_layout(template=AMR_TEMPLATE, height=300,
                               margin=dict(t=10,b=80,l=100,r=10),
                               xaxis=dict(tickangle=-40,tickfont_size=10))
            st.plotly_chart(fig6, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="section-header">MDR Rate by Environment Type</div>', unsafe_allow_html=True)
                env_mdr = env.groupby("host_species")["is_mdr"].mean().sort_values(ascending=False).reset_index()
                fig7 = px.bar(env_mdr, x="host_species", y="is_mdr",
                              color="is_mdr", color_continuous_scale="Purples",
                              text=env_mdr["is_mdr"].map(lambda x: f"{x:.1%}"),
                              labels={"is_mdr":"MDR Rate","host_species":""})
                fig7.update_traces(textposition="outside")
                fig7.update_layout(template=AMR_TEMPLATE, height=300,
                                   yaxis_tickformat=".0%", coloraxis_showscale=False)
                st.plotly_chart(fig7, use_container_width=True)
            with c2:
                st.markdown('<div class="section-header">Colistin Resistance — Last-Resort Threat</div>', unsafe_allow_html=True)
                col_r = df.groupby("host_type")["Colistin"].mean().reset_index()
                fig8 = px.bar(col_r, x="host_type", y="Colistin",
                              color="host_type", color_discrete_map=HOST_TYPE_COLORS,
                              text=col_r["Colistin"].map(lambda x: f"{x:.1%}"),
                              labels={"Colistin":"Colistin Resistance","host_type":""})
                fig8.update_traces(textposition="outside")
                fig8.update_layout(template=AMR_TEMPLATE, height=300,
                                   yaxis_tickformat=".0%", showlegend=False)
                st.plotly_chart(fig8, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EXPLORATORY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Exploratory Analysis":
    st.markdown("""
    <h1 style="margin:0;font-size:1.75rem;font-weight:800;color:#f1f5f9">Exploratory Data Analysis</h1>
    <p style="color:#475569;font-size:.88rem">Deep-dive into resistance patterns across all host categories</p>
    <hr style="border-color:#0f2035;margin:1rem 0">
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🧫 Antibiogram","👤 Demographics","📋 Raw Data"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1: sel_sp = st.selectbox("Species", sorted(df["species"].unique()))
        with c2: ab_class_filter = st.multiselect("Antibiotic classes",
                                                   list(ANTIBIOTIC_CLASSES.keys()),
                                                   default=list(ANTIBIOTIC_CLASSES.keys()))

        filtered_abs = [ab for cls, lst in ANTIBIOTIC_CLASSES.items()
                        if cls in ab_class_filter for ab in lst if ab in ALL_ANTIBIOTICS]
        sp_df = df[df["species"] == sel_sp]
        rr = sp_df[filtered_abs].mean().sort_values(ascending=False).reset_index()
        rr.columns = ["antibiotic","resistance_rate"]
        rr["class"] = rr["antibiotic"].map(
            lambda ab: next((cls for cls, lst in ANTIBIOTIC_CLASSES.items() if ab in lst), "Other"))
        fig = px.bar(rr, x="antibiotic", y="resistance_rate",
                     color="class", text=rr["resistance_rate"].map(lambda x: f"{x:.1%}"),
                     title=f"Antibiogram — {sel_sp}",
                     labels={"resistance_rate":"Resistance","antibiotic":""})
        fig.add_hline(y=0.5, line_dash="dash", line_color="#f43f5e",
                      annotation_text="50%", annotation_position="top right")
        fig.update_traces(textposition="outside", textfont_size=10)
        fig.update_layout(template=AMR_TEMPLATE, height=400,
                          xaxis_tickangle=-40,
                          legend=dict(orientation="h",y=-0.28))
        st.plotly_chart(fig, use_container_width=True)

        # Resistance rate by host × species
        st.markdown('<div class="section-header">Resistance Rate Distribution by Host Category</div>', unsafe_allow_html=True)
        fig2 = px.violin(df, x="host_type", y="resistance_rate",
                         color="host_type", color_discrete_map=HOST_TYPE_COLORS,
                         box=True, points=False,
                         labels={"resistance_rate":"Resistance Rate","host_type":""})
        fig2.update_layout(template=AMR_TEMPLATE, height=340, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        df["MDR Status"] = df["is_mdr"].map({0:"Non-MDR",1:"MDR"})
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">Age Distribution by MDR Status (Animals & Humans)</div>', unsafe_allow_html=True)
            age_df = df[df["host_age"] >= 0]
            fig3 = px.histogram(age_df, x="host_age", color="MDR Status",
                                nbins=30, barmode="overlay", opacity=0.75,
                                color_discrete_map={"MDR":"#f43f5e","Non-MDR":"#14b8a6"},
                                labels={"host_age":"Age"})
            fig3.update_layout(template=AMR_TEMPLATE, height=300)
            st.plotly_chart(fig3, use_container_width=True)
        with c2:
            st.markdown('<div class="section-header">Prior Antibiotic Exposure vs MDR</div>', unsafe_allow_html=True)
            exp_df = df.groupby(["prior_antibiotic_exposure","MDR Status"]).size().reset_index(name="count")
            exp_df["Exposure"] = exp_df["prior_antibiotic_exposure"].map({0:"None",1:"Prior exposure"})
            fig4 = px.bar(exp_df, x="Exposure", y="count", color="MDR Status",
                          barmode="group",
                          color_discrete_map={"MDR":"#f43f5e","Non-MDR":"#14b8a6"})
            fig4.update_layout(template=AMR_TEMPLATE, height=300)
            st.plotly_chart(fig4, use_container_width=True)

        st.markdown('<div class="section-header">Resistance Rate by Host Category × Antibiotic Class</div>', unsafe_allow_html=True)
        ab_class_avgs = {}
        for cls, abs_list in ANTIBIOTIC_CLASSES.items():
            ab_class_avgs[cls] = df[[ab for ab in abs_list if ab in df.columns]].mean(axis=1)
        class_df = pd.DataFrame(ab_class_avgs)
        class_df["host_type"] = df["host_type"].values
        class_melt = class_df.melt(id_vars="host_type", var_name="Class", value_name="Resistance")
        fig5 = px.box(class_melt, x="Class", y="Resistance", color="host_type",
                      color_discrete_map=HOST_TYPE_COLORS,
                      labels={"host_type":"Host Category"})
        fig5.update_layout(template=AMR_TEMPLATE, height=380,
                           xaxis_tickangle=-30,
                           legend=dict(orientation="h",y=-0.25))
        st.plotly_chart(fig5, use_container_width=True)

    with tab3:
        st.markdown(f'<div class="section-header">Dataset Preview ({len(df):,} records)</div>', unsafe_allow_html=True)
        display = ["isolate_id","host_species","host_type","species","gram_stain",
                   "country","host_sex","host_age","zoonotic","zoonotic_risk",
                   "prior_antibiotic_exposure","esbl","mrsa","vre",
                   "mdr_phenotype","is_mdr","resistance_rate"] + ALL_ANTIBIOTICS[:8]
        st.dataframe(df[[c for c in display if c in df.columns]].head(200),
                     use_container_width=True, height=400)
        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Download full dataset (CSV)", csv, "amr_one_health.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ML MODEL
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 ML Model":
    st.markdown("""
    <h1 style="margin:0;font-size:1.75rem;font-weight:800;color:#f1f5f9">Machine Learning Model</h1>
    <p style="color:#475569;font-size:.88rem">Random Forest — One Health MDR Prediction</p>
    <hr style="border-color:#0f2035;margin:1rem 0">
    """, unsafe_allow_html=True)

    model, feature_cols, metrics = get_model()

    if model is None:
        st.info("🔧 Model not trained yet. Click below to train.")
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training (~30s)…"):
                from utils.train_model import train
                _, metrics = train(use_antibiogram=True)
                st.cache_resource.clear()
            st.success("✅ Done!")
            st.rerun()
    else:
        rep     = metrics["classification_report"]
        roc_auc = metrics["roc_auc"]

        k1,k2,k3,k4 = st.columns(4)
        with k1: st.metric("ROC-AUC",  f'{roc_auc:.4f}', f'CV {metrics["cv_auc_mean"]:.3f}')
        with k2: st.metric("Precision (MDR)", f'{rep["1"]["precision"]:.3f}')
        with k3: st.metric("Recall (MDR)",    f'{rep["1"]["recall"]:.3f}')
        with k4: st.metric("F1-Score (MDR)",  f'{rep["1"]["f1-score"]:.3f}')

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2 = st.columns([1.2,1])

        with c1:
            st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = np.array(metrics["confusion_matrix"])
            fig = go.Figure(go.Heatmap(
                z=cm, x=["Non-MDR","MDR"], y=["Non-MDR","MDR"],
                colorscale="Blues", text=cm, texttemplate="%{text}",
                textfont=dict(size=22,color="white"), showscale=False,
            ))
            fig.update_layout(template=AMR_TEMPLATE, height=320)
            fig.update_xaxes(title="Predicted")
            fig.update_yaxes(title="Actual", autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
            fpr = metrics.get("roc_fpr", [])
            tpr = metrics.get("roc_tpr", [])
            if not fpr:
                from sklearn.metrics import roc_curve as rc
                fpr,tpr,_ = rc(metrics["y_test"], metrics["y_proba"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     line=dict(color="#3b82f6",width=2.5),
                                     name=f"AUC={roc_auc:.3f}"))
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1], mode="lines",
                                     line=dict(color="#334155",dash="dash"),name="Random"))
            fig.update_layout(template=AMR_TEMPLATE, height=320,
                              xaxis_title="FPR", yaxis_title="TPR",
                              legend=dict(x=0.6,y=0.1))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Top 20 Most Important Features</div>', unsafe_allow_html=True)
        fi = pd.Series(metrics["feature_importance"]).sort_values(ascending=False).head(20)
        fig2 = go.Figure(go.Bar(
            x=fi.values[::-1], y=fi.index[::-1], orientation="h",
            marker=dict(color=fi.values[::-1], colorscale="Teal", showscale=False),
            text=[f"{v:.4f}" for v in fi.values[::-1]], textposition="outside",
        ))
        fig2.update_layout(template=AMR_TEMPLATE, height=520,
                           margin=dict(l=280,r=60,t=10,b=10),
                           xaxis_title="Importance")
        st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — AMR PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🧬 AMR Predictor":
    st.markdown("""
    <h1 style="margin:0;font-size:1.75rem;font-weight:800;color:#f1f5f9">Live One Health AMR Predictor</h1>
    <p style="color:#475569;font-size:.88rem">Enter isolate characteristics for any host species to predict MDR probability</p>
    <hr style="border-color:#0f2035;margin:1rem 0">
    """, unsafe_allow_html=True)

    model, feature_cols, metrics = get_model()
    if model is None:
        st.warning("⚠️ Model not trained. Go to **🤖 ML Model** first.")
        st.stop()

    with st.form("pred_form"):
        st.markdown("### 🧬 Host & Isolate Information")
        r1, r2, r3 = st.columns(3)

        with r1:
            p_host    = st.selectbox("Host Species", list(HOST_CATEGORIES.keys()))
            host_info = HOST_CATEGORIES[p_host]
            st.markdown(f'<span class="oh-pill">{host_info["icon"]} {host_info["type"]}</span>', unsafe_allow_html=True)
            p_country = st.selectbox("Country", COUNTRIES)
            p_year    = st.slider("Year", 2015, 2024, 2023)

        with r2:
            # Filter bacteria compatible with this host
            compat_bact = [sp for sp, info in BACTERIA.items() if p_host in info["hosts"]]
            p_species   = st.selectbox("Bacterial Species", compat_bact)
            bact_info   = BACTERIA[p_species]
            st.markdown(
                f'<span class="badge badge-host">{bact_info["gram"]}</span>&nbsp;'
                f'<span class="badge badge-{"mdr" if bact_info["zoonotic_risk"] in ["High","Very High"] else "sus"}">'
                f'Zoonotic: {bact_info["zoonotic_risk"]}</span>',
                unsafe_allow_html=True
            )
            p_sex     = st.selectbox("Sex", ["Male","Female","Unknown","N/A"])
            p_age     = st.slider("Age (years, -1 = N/A)", -1, 30, 5)

        with r3:
            p_prior_ab   = st.checkbox("Prior antibiotic exposure")
            p_intensive  = st.checkbox("ICU / intensive farming")
            p_esbl       = st.checkbox("ESBL Producer")
            p_carbapenem = st.checkbox("Carbapenemase Producer")
            p_mrsa       = st.checkbox("MRSA")
            p_vre        = st.checkbox("VRE")

        st.markdown("---")
        st.markdown("### 💊 Antibiogram (✓ = Resistant)")
        ab_inputs = {}
        for cls_name, abs_list in ANTIBIOTIC_CLASSES.items():
            st.markdown(f"**{cls_name}**")
            cols = st.columns(min(len(abs_list), 4))
            for i, ab in enumerate(abs_list):
                with cols[i % len(cols)]:
                    ab_inputs[ab] = int(st.checkbox(ab, key=f"ab_{ab}"))

        submitted = st.form_submit_button("🔮 Predict MDR Status", type="primary", use_container_width=True)

    if submitted:
        input_dict = {
            "host_species":             p_host,
            "host_type":                host_info["type"],
            "species":                  p_species,
            "gram_stain":               bact_info["gram"],
            "family":                   bact_info["family"],
            "country":                  p_country,
            "host_sex":                 p_sex,
            "host_age":                 p_age,
            "zoonotic_risk":            bact_info["zoonotic_risk"],
            "zoonotic_score":           ZOONOTIC_RISK_SCORE[bact_info["zoonotic_risk"]],
            "prior_antibiotic_exposure": int(p_prior_ab),
            "intensive_care_or_farming": int(p_intensive),
            "esbl":                     int(p_esbl),
            "carbapenemase":            int(p_carbapenem),
            "mrsa":                     int(p_mrsa),
            "vre":                      int(p_vre),
            "year":                     p_year,
            **ab_inputs,
        }
        for col in feature_cols:
            if col not in input_dict:
                input_dict[col] = 0

        X_in       = pd.DataFrame([input_dict])[feature_cols]
        pred_proba = model.predict_proba(X_in)[0][1]
        pred_label = int(pred_proba >= 0.5)

        col_res, col_gauge = st.columns([1,1])

        with col_res:
            css_class = "pred-mdr" if pred_label else "pred-sus"
            icon      = "⚠️ MDR PREDICTED" if pred_label else "✅ NON-MDR"
            st.markdown(f"""
            <div class="pred-box {css_class}">
                {icon}<br>
                <span style="font-size:.85rem;opacity:.8">P(MDR) = {pred_proba:.1%}</span>
            </div>""", unsafe_allow_html=True)

            if pred_label:
                st.error("**Alert:** MDR profile detected. Consider specialist consultation and last-resort antibiotics.")
            else:
                st.success("**Non-MDR profile.** Standard therapy likely effective. Verify with laboratory culture.")

            st.markdown("**Isolate Summary**")
            n_resistant = sum(ab_inputs.values())
            summary = {
                f"{host_info['icon']} Host": f"{p_host} ({host_info['type']})",
                "Species": p_species, "Gram": bact_info["gram"],
                "Zoonotic Risk": bact_info["zoonotic_risk"],
                "Country": p_country, "Year": p_year,
                "Prior Antibiotics": "Yes" if p_prior_ab else "No",
                "ESBL/MRSA/VRE": f"{'ESBL ' if p_esbl else ''}{'MRSA ' if p_mrsa else ''}{'VRE' if p_vre else ''}" or "None",
                "Resistant to": f"{n_resistant}/{len(ab_inputs)} antibiotics tested",
            }
            for k, v in summary.items():
                st.markdown(f"- **{k}:** {v}")

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred_proba * 100,
                title=dict(text="MDR Probability (%)", font_size=14),
                number=dict(suffix="%", font_size=36),
                delta=dict(reference=50,
                           increasing_color="#f43f5e",
                           decreasing_color="#14b8a6"),
                gauge=dict(
                    axis=dict(range=[0,100], tickfont_size=11),
                    bar=dict(color="#f43f5e" if pred_label else "#14b8a6"),
                    steps=[
                        dict(range=[0,30],   color="rgba(20,184,166,0.13)"),
                        dict(range=[30,60],  color="rgba(245,158,11,0.13)"),
                        dict(range=[60,100], color="rgba(244,63,94,0.13)"),
                    ],
                    threshold=dict(line=dict(color="#f59e0b",width=3),
                                   thickness=0.8, value=50),
                ),
            ))
            fig.update_layout(template=AMR_TEMPLATE, height=320,
                              margin=dict(t=40,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)

            # One Health zoonotic alert
            if bact_info["zoonotic"] and bact_info["zoonotic_risk"] in ["High","Very High"]:
                st.markdown(f"""
                <div style="background:#f59e0b11;border:1px solid #f59e0b33;border-radius:10px;padding:.9rem;margin-top:.5rem">
                    <b style="color:#f59e0b">🔗 One Health Zoonotic Alert</b><br>
                    <span style="font-size:.78rem;color:#94a3b8">
                    <b>{p_species}</b> is a <b>{bact_info["zoonotic_risk"]}</b> zoonotic pathogen.
                    This isolate ({p_host}) may pose transmission risks to humans and other hosts.
                    Implement barrier precautions and notify public health authorities if MDR confirmed.
                    </span>
                </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0d1b2e;border:1px solid #f59e0b33;border-radius:10px;
                padding:1rem;margin-top:1.5rem">
        <b style="color:#f59e0b">⚠️ Educational Disclaimer</b><br>
        <span style="font-size:.78rem;color:#64748b">
        Synthetic data — academic portfolio use only. Not for clinical or veterinary decision-making.
        All treatment decisions must be guided by certified laboratory results and specialist expertise.
        </span>
    </div>""", unsafe_allow_html=True)
