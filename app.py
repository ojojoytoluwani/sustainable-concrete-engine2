# ==============================================================================
# SUSTAINABLE CONCRETE ENGINE: GENERATIVE DESIGN & INFORMATICS PLATFORM
# Standard Compliance: BS EN 1992-1-1 (EC2), BS EN 206, BS 8500-1/2, EN 15804
# ==============================================================================
import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

# 1. Page Configuration & Custom UI Styling
st.set_page_config(
    page_title="Sustainable Concrete Engine | Agro-Waste Informatics",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #003366;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #495057;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 4px solid #0055b3;
        padding: 0.8rem 1rem;
        border-radius: 4px;
        margin-bottom: 0.6rem;
    }
    .audit-pass {
        color: #0f5132;
        background-color: #d1e7dd;
        border: 1px solid #badbcc;
        padding: 0.75rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .audit-fail {
        color: #842029;
        background-color: #f8d7da;
        border: 1px solid #f5c2c7;
        padding: 0.75rem;
        border-radius: 4px;
        font-weight: 600;
    }
    /* Prevent metric value clipping and scale font */
    div[data-testid="stMetricValue"] > div {
        font-size: 1.28rem !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
    }
    /* Text wrapping in selectboxes */
    div[data-baseweb="select"] * {
        white-space: normal !important;
        word-break: break-word !important;
        line-height: 1.35 !important;
    }
    div[data-baseweb="popover"] li {
        padding-top: 8px !important;
        padding-bottom: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# 2. Environmental Inventory Factors & Engineering Constants
EMISSION_FACTORS = {
    "Cement": 0.912,
    "RHA": 0.05,
    "POFA": 0.035,
    "SCBA": 0.035,
    "Water": 0.0002,
    "Fine_Agg": 0.0051,
    "Coarse_Agg": 0.0075,
    "SP": 1.25
}

OPC_REF = {
    "Cement": 300.0,
    "Fine_Agg": 865.2,
    "Coarse_Agg": 950.0,
    "Water": 135.0,
    "SP": 0.00,
    "fc": 35.97,
    "GWP": 285.17,
    "Ci": 9.91
}

DEFAULT_OXIDES = {
    "RHA": {
        "SiO2": 92.26,
        "Al2O3": 0.39,
        "Fe2O3": 0.32,
        "CaO": 0.80,
        "K2O": 1.40,
        "LOI": 0.51
    },
    "POFA": {
        "SiO2": 57.71,
        "Al2O3": 4.56,
        "Fe2O3": 3.30,
        "CaO": 6.55,
        "K2O": 8.27,
        "LOI": 10.30
    },
    "SCBA": {
        "SiO2": 65.00,
        "Al2O3": 4.82,
        "Fe2O3": 4.01,
        "CaO": 4.04,
        "K2O": 4.00,
        "LOI": 4.56
    }
}

GRINDING_DEFINITIONS = {
    "Low: ash is raw, unground, or sieved only": 1,
    "Medium: ash underwent short manual or disc milling": 2,
    "High (> 10 µm): ash ball-milled (particle size > 10 micrometers)": 3,
    "Ultra High (< 10 µm): ash ball-milled (particle size < 10 micrometers)": 4
}

# 3. Model & Data Artifact Ingestion Pipeline (Cached with Base_Score Interceptor)
@st.cache_resource(show_spinner=False)
def load_computational_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "xgb_final_784.joblib")
    
    if not os.path.exists(model_path):
        model_path = "/content/drive/MyDrive/MSc_Concrete_Strength/xgb_final_784.joblib"
    
    model = joblib.load(model_path)
    
    # Intercept and clean bracketed base_score '[3.032346E1]' for SHAP compatibility
    orig_json_loads = json.loads
    def patched_json_loads(s, *args, **kwargs):
        obj = orig_json_loads(s, *args, **kwargs)
        if isinstance(obj, dict) and "learner" in obj:
            lp = obj["learner"].get("learner_model_param", {})
            if "base_score" in lp and isinstance(lp["base_score"], str):
                cleaned = lp["base_score"].strip("[]")
                try:
                    float(cleaned)
                    lp["base_score"] = cleaned
                except ValueError:
                    pass
        return obj

    try:
        json.loads = patched_json_loads
        explainer = shap.TreeExplainer(model)
    finally:
        json.loads = orig_json_loads
    
    pareto_data = {}
    for scm in ["RHA", "POFA", "SCBA"]:
        fpath = os.path.join(base_dir, f"pareto_optimal_mixes_{scm}_with_carbon.csv")
        if not os.path.exists(fpath):
            fpath = f"/content/drive/MyDrive/MSc_Concrete_Strength/pareto_results/pareto_optimal_mixes_{scm}_with_carbon.csv"
        if os.path.exists(fpath):
            pareto_data[scm] = pd.read_csv(fpath)
            
    dec_path = os.path.join(base_dir, "pareto_decision_matrix_with_carbon.csv")
    if not os.path.exists(dec_path):
        dec_path = "/content/drive/MyDrive/MSc_Concrete_Strength/pareto_results/pareto_decision_matrix_with_carbon.csv"
    
    df_decision = pd.read_csv(dec_path) if os.path.exists(dec_path) else pd.DataFrame()

    return model, explainer, pareto_data, df_decision

try:
    model, explainer, pareto_results, df_decision_matrix = load_computational_artifacts()
    artifacts_ready = True
except Exception as e:
    artifacts_ready = False
    load_error_msg = str(e)

# Range of each batch mass in the 778-record database, by material (kg/m³)
DATA_RANGES = {
    "RHA":  {"Cement": (198.8, 428.4), "Ash": (17.4, 142.8), "Water": (153.0, 228.0), "Fine aggregate": (464.0, 900.0), "Coarse aggregate": (927.0, 1394.1), "Superplasticiser": (0.0, 11.8)},
    "POFA": {"Cement": (60.0, 433.0), "Ash": (5.3, 266.0), "Water": (103.5, 285.0), "Fine aggregate": (200.0, 805.0), "Coarse aggregate": (400.0, 1371.4), "Superplasticiser": (0.0, 16.5)},
    "SCBA": {"Cement": (183.0, 377.9), "Ash": (15.4, 183.0), "Water": (150.0, 210.1), "Fine aggregate": (546.0, 1010.0), "Coarse aggregate": (932.0, 1320.0), "Superplasticiser": (0.8, 5.6)},
    "None": {"Cement": (100.0, 500.0), "Water": (103.5, 250.0), "Fine aggregate": (200.0, 1010.0), "Coarse aggregate": (400.0, 1394.1), "Superplasticiser": (0.0, 16.5)},
}

# 4. Eurocode 2 & BS 8500 Auditing Functions
def evaluate_structural_compliance(cement, scm, water, fine, coarse, sp, scm_type):
    total_binder = cement + scm
    wb_ratio = water / total_binder if total_binder > 0 else 999.0
    fresh_density = cement + scm + water + fine + coarse + sp
    rep_pct = (scm / total_binder) * 100.0 if total_binder > 0 else 0.0
    
    violations = []
    warnings = []

    # Limits are those of the optimisation (Chapter 3, Section 3.9.3), not code requirements.
    if cement < 180.0:
        violations.append(
            f"Cement ({cement:.1f} kg/m³) is below the 180 kg/m³ clinker floor adopted in this study "
            "to retain a portlandite reserve for reinforcement. This is an author-imposed limit, "
            "not a requirement of BS EN 206 or BS 8500, and it does not apply to plain concrete."
        )
    if total_binder < 300.0 or total_binder > 480.0:
        warnings.append(
            f"Binder content ({total_binder:.1f} kg/m³) is outside the 300–480 kg/m³ range used in the optimisation."
        )
    if wb_ratio < 0.32 or wb_ratio > 0.48:
        warnings.append(
            f"Water–binder ratio ({wb_ratio:.3f}) is outside the 0.32–0.48 range used in the optimisation."
        )
    if fresh_density < 2250.0 or fresh_density > 2450.0:
        warnings.append(
            f"Fresh density ({fresh_density:.1f} kg/m³) is outside the 2250–2450 kg/m³ range used in the optimisation."
        )
    if scm > 0 and (rep_pct < 5.0 or rep_pct > 50.0):
        warnings.append(f"Ash fraction ({rep_pct:.1f}%) is outside the 5–50% range used in the optimisation.")

    # Range of the training data for this material (prediction is unreliable outside it)
    lo_hi = DATA_RANGES.get(scm_type if scm > 0 else "None", {})
    vals = {"Cement": cement, "Ash": scm, "Water": water, "Fine aggregate": fine,
            "Coarse aggregate": coarse, "Superplasticiser": sp}
    for k, v in vals.items():
        if k in lo_hi and not (lo_hi[k][0] <= v <= lo_hi[k][1]):
            warnings.append(
                f"{k} ({v:.1f} kg/m³) lies outside the range of the training data for this material "
                f"({lo_hi[k][0]:.0f}–{lo_hi[k][1]:.0f} kg/m³). The prediction is an extrapolation and is unreliable."
            )

    exposures = []
    if wb_ratio <= 0.65 and total_binder >= 260: exposures.append("XC1 (Dry Interior)")
    if wb_ratio <= 0.60 and total_binder >= 280: exposures.append("XC2 (Foundation / Wet Soil)")
    if wb_ratio <= 0.55 and total_binder >= 300: exposures.append("XC3/XC4 (External Facades)")
    if wb_ratio <= 0.45 and total_binder >= 340: exposures.append("XD1/XS1 (Chlorides / Seawater Spray)")

    cement_type = ("Plain CEM I" if rep_pct <= 0.0 else
                   f"Binary blend: CEM I + {rep_pct:.1f}% {scm_type} (not an EN 197-1 cement type)")

    scm_ef = EMISSION_FACTORS.get(scm_type, 0.0400)
    gwp = (
        cement * EMISSION_FACTORS["Cement"] +
        scm * scm_ef +
        water * EMISSION_FACTORS["Water"] +
        fine * EMISSION_FACTORS["Fine_Agg"] +
        coarse * EMISSION_FACTORS["Coarse_Agg"] +
        sp * EMISSION_FACTORS["SP"]
    )
    abatement_pct = ((OPC_REF["GWP"] - gwp) / OPC_REF["GWP"]) * 100.0

    return {
        "is_compliant": len(violations) == 0,
        "violations": violations,
        "warnings": warnings,
        "wb_ratio": wb_ratio,
        "density": fresh_density,
        "rep_pct": rep_pct,
        "total_binder": total_binder,
        "cement_type": cement_type,
        "exposures": exposures,
        "GWP": gwp,
        "Abatement_Pct": abatement_pct
    }

def map_eurocode_capacity(f_cm: float):
    f_ck_cyl = max(0.0, 0.80 * f_cm - 8.0)  # model predicts 150 mm cube strength
    f_ck_cube = f_ck_cyl / 0.80
    
    classes = [
        (50, 60, "C50/60"), (45, 55, "C45/55"), (40, 50, "C40/50"),
        (35, 45, "C35/45"), (32, 40, "C32/40"), (30, 37, "C30/37"),
        (25, 30, "C25/30"), (20, 25, "C20/25"), (16, 20, "C16/20"),
        (12, 15, "C12/15")
    ]
    designation = "Sub-Structural (< C12/15)"
    for cyl_lim, cube_lim, code in classes:
        if f_ck_cyl >= cyl_lim:
            designation = code
            break
            
    return f_ck_cyl, f_ck_cube, designation

def same_class_cem1_gwp(ec2_class: str):
    """GWP of the lowest-carbon CEM I mix on the Pareto front that reaches the same class.
    Returns None below C20/25, where the leanest CEM I mix is already over-strength,
    so no like-for-like saving exists."""
    try:
        ref = df_decision_matrix[(df_decision_matrix["SCM_Type"] == "CEM I") &
                                 (df_decision_matrix["Target_Class"] == ec2_class)]
        if ec2_class in ("C20/25", "C25/30") and not ref.empty:
            return float(ref["GWP_kgCO2e_m3"].iloc[0])
    except Exception:
        pass
    return None

def saving_text(gwp: float, ec2_class: str) -> str:
    if ec2_class == "C25/30":
        return ("n/a: no CEM I mix on the front reaches C25/30, so there is no like-for-like reference "
                "(the mix lies near the model's predictive ceiling)")
    ref = same_class_cem1_gwp(ec2_class)
    if ref is None:
        return "n/a (no like-for-like CEM I mix below C20/25)"
    return f"{(ref - gwp) / ref * 100:.1f}% vs CEM I {ec2_class} ({ref:.1f})"

def saving_short(gwp: float, ec2_class: str) -> str:
    t = saving_text(gwp, ec2_class)
    return t.split(" vs ")[0] if t[0].isdigit() or t[0] == "-" else "n/a"

# 5. Application UI & Tabs
st.markdown('<div class="main-title">Sustainable Concrete Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Generative Mix Design, Eurocode 2 Structural Calibration & Mechanistic SHAP Auditing '
    'for Tropical Agro-Waste Concrete (RHA, POFA, SCBA)'
    '</div>',
    unsafe_allow_html=True
)

if not artifacts_ready:
    st.error(f"Initialization Failed: Could not load core artifacts. Error: {load_error_msg}")
    st.stop()

st.info("Screening tool. Predictions carry a typical error of about 5.7 MPa (150 mm cube basis) on studies not seen in training, and proposed mixes are candidates for trial batching, not specifications. Strength classes are assigned from point predictions.")
tab_forward, tab_generative, tab_matrix = st.tabs([
    "🔬 Forward Prediction & SHAP Audit",
    "⚡ Inverse Generative Designer (NSGA-II)",
    "📋 Eurocode Decision Matrix & LCA Scorecard"
])

# --- TAB 1: FORWARD PREDICTION ---
with tab_forward:
    st.markdown("### Strength prediction and model attribution")

    if "cement_val" not in st.session_state:
        st.session_state.cement_val = 300.0
    if "scm_mass_val" not in st.session_state:
        st.session_state.scm_mass_val = 100.0
    if "scm_rep_pct" not in st.session_state:
        st.session_state.scm_rep_pct = 25.0

    def update_from_mass():
        tot = st.session_state.cement_val + st.session_state.scm_mass_val
        st.session_state.scm_rep_pct = round((st.session_state.scm_mass_val / tot) * 100.0, 1) if tot > 0 else 0.0

    def update_from_pct():
        r = st.session_state.scm_rep_pct
        if r < 100.0:
            st.session_state.scm_mass_val = round(st.session_state.cement_val * (r / (100.0 - r)), 1)

    def update_from_cement():
        r = st.session_state.scm_rep_pct
        if r < 100.0:
            st.session_state.scm_mass_val = round(st.session_state.cement_val * (r / (100.0 - r)), 1)

    col_mix, col_chem, col_pred = st.columns([1.1, 1.1, 1.4], gap="medium")

    with col_mix:
        st.markdown("#### 1. Batch Proportions (kg/m³)")
        scm_type = st.selectbox("Agro-Waste Material:", ["RHA", "POFA", "SCBA"], index=0)
        
        st.number_input(
            "Portland Cement (CEM I):", 
            min_value=120.0, max_value=600.0, step=10.0,
            key="cement_val", 
            on_change=update_from_cement
        )
        
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            st.number_input(
                f"{scm_type} Mass (kg/m³):",
                min_value=0.0, max_value=350.0, step=5.0,
                key="scm_mass_val",
                on_change=update_from_mass
            )
        with c_m2:
            st.number_input(
                "Replacement (% SCM):",
                min_value=0.0, max_value=55.0, step=0.5,
                key="scm_rep_pct",
                on_change=update_from_pct
            )

        cement_val = st.session_state.cement_val
        scm_val = st.session_state.scm_mass_val

        water_val = st.number_input("Water:", min_value=100.0, max_value=290.0, value=160.0, step=5.0)
        fine_val = st.number_input("Fine aggregate:", min_value=200.0, max_value=1050.0, value=750.0, step=10.0)
        coarse_val = st.number_input("Coarse aggregate:", min_value=400.0, max_value=1400.0, value=1050.0, step=10.0)
        sp_val = st.number_input("Superplasticizer (SP):", min_value=0.0, max_value=15.0, value=2.5, step=0.5)

    with col_chem:
        st.markdown("#### 2. Curing, Milling & Chemistry")
        age_val = st.slider("Curing Age (Days):", min_value=3, max_value=90, value=28, step=1)
        control_str_val = st.number_input(
            "Reference Lab 28d Control Strength (MPa):", 
            min_value=15.0, max_value=50.0, value=35.0, step=0.5,
            help="28-day 150 mm cube strength of a plain CEM I control cast with the same materials. The training data span 16.8–48.0 MPa."
        )
        
        selected_grind_label = st.selectbox(
            "Mechanical Grinding Intensity:",
            options=list(GRINDING_DEFINITIONS.keys()),
            index=1,
            help="Mechanical activation level and particle size classification."
        )
        grind_val = GRINDING_DEFINITIONS[selected_grind_label]
        st.caption(f"📌 **Active Definition:** Level {grind_val} — {selected_grind_label}")
        
        ox_defaults = DEFAULT_OXIDES[scm_type]
        sio2_val = st.number_input("SiO₂ Content:", value=ox_defaults["SiO2"], step=0.5)
        al2o3_val = st.number_input("Al₂O₃ Content:", value=ox_defaults["Al2O3"], step=0.2)
        fe2o3_val = st.number_input("Fe₂O₃ Content:", value=ox_defaults["Fe2O3"], step=0.2)
        cao_val = st.number_input("CaO Content:", value=ox_defaults["CaO"], step=0.2)
        k2o_val = st.number_input("K₂O Content:", value=ox_defaults["K2O"], step=0.2)
        loi_val = st.number_input("Loss on Ignition (LOI):", value=ox_defaults["LOI"], step=0.5)
        other_ox_val = st.number_input(
            "Other oxides:", value={"RHA": 1.21, "POFA": 9.09, "SCBA": 4.75}[scm_type], step=0.5,
            help="Pre-filled with the database median used in the optimisation (Table 3.7). "
                 "Median profiles do not sum to 100%, so this is not computed as a balance.")

    input_dict = {
        "Cement_Content": cement_val,
        "SCM_Mass": scm_val,
        "Water_Content": water_val,
        "Fine_Aggregate": fine_val,
        "Coarse_Aggregate": coarse_val,
        "Superplasticizer_Mass": sp_val,
        "Curing_Age": age_val,
        "Control_Strength_28d": control_str_val,
        "Grinding_Intensity_enc": grind_val if scm_val > 0 else 0,
        "SCM_SiO2": sio2_val if scm_val > 0 else 0.0,
        "SCM_Al2O3": al2o3_val if scm_val > 0 else 0.0,
        "SCM_Fe2O3": fe2o3_val if scm_val > 0 else 0.0,
        "SCM_CaO": cao_val if scm_val > 0 else 0.0,
        "SCM_K2O": k2o_val if scm_val > 0 else 0.0,
        "SCM_LOI": loi_val if scm_val > 0 else 0.0,
        "SCM_Other_Oxides": other_ox_val if scm_val > 0 else 0.0,
        "SCM_Type_None": 1 if scm_val == 0 else 0,
        "SCM_Type_POFA": 1 if (scm_type == "POFA" and scm_val > 0) else 0,
        "SCM_Type_RHA": 1 if (scm_type == "RHA" and scm_val > 0) else 0,
        "SCM_Type_SCBA": 1 if (scm_type == "SCBA" and scm_val > 0) else 0
    }
    
    df_input = pd.DataFrame([input_dict])
    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in df_input.columns:
                df_input[col] = 0.0
        df_input = df_input[model.feature_names_in_]

    audit_res = evaluate_structural_compliance(cement_val, scm_val, water_val, fine_val, coarse_val, sp_val, scm_type)
    pred_fcm = float(model.predict(df_input)[0])
    f_ck_cyl, f_ck_cube, ec2_class = map_eurocode_capacity(pred_fcm)
    eco_index_ci = audit_res["GWP"] / (0.80 * pred_fcm) if pred_fcm > 0 else 999.0

    with col_pred:
        st.markdown("#### 3. Eurocode 2 Designations")
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.9rem; color:#666;">EUROCODE 2 DESIGNATION</span><br>
            <span style="font-size:1.8rem; font-weight:700; color:#0055b3;">{ec2_class}</span><br>
            <span style="font-size:0.85rem; color:#333;">Composition limits met (indicative, BS EN 206 Annex F; strength class not checked): {', '.join(audit_res['exposures'][:2]) if audit_res['exposures'] else 'none'}</span>
        </div>
        """, unsafe_allow_html=True)
        
        s1, s2, s3 = st.columns(3)
        s1.metric("Predicted cube $f_{cu}$", f"{pred_fcm:.1f} MPa")
        s2.metric("Char. f_ck,cyl", f"{f_ck_cyl:.1f} MPa")
        s3.metric("Char. f_ck,cube", f"{f_ck_cube:.1f} MPa")

        st.markdown("#### 4. Environmental Accounting (EN 15804)")
        e1, e2, e3 = st.columns(3)
        e1.metric("Embodied Carbon", f"{audit_res['GWP']:.1f} kg CO₂e/m³")
        e2.metric("Carbon Abated (same class)", saving_short(audit_res["GWP"], ec2_class))
        e3.metric("Eco-Efficiency (Ci)", f"{eco_index_ci:.2f}")

        st.markdown("#### 5. Composition checks (study limits)")
        if audit_res["is_compliant"]:
            st.markdown(f'<div class="audit-pass">✓ Meets the author-imposed clinker floor. ({audit_res["cement_type"]})</div>', unsafe_allow_html=True)
        else:
            fail_text = "<br>• ".join(audit_res["violations"])
            st.markdown(f'<div class="audit-fail">✗ Outside the study limits:<br>• {fail_text}</div>', unsafe_allow_html=True)

        if audit_res["warnings"]:
            for w in audit_res["warnings"]: st.warning(w)

    st.markdown("---")
    st.markdown("### Model attribution: TreeSHAP decomposition (describes the model, not chemical mechanisms)")
    shap_exp = explainer(df_input)
    shap.plots.waterfall(shap_exp[0], max_display=10, show=False)
    fig_shap = plt.gcf()
    st.pyplot(fig_shap)
    plt.close(fig_shap)

# --- TAB 2: INVERSE GENERATIVE DESIGNER ---
with tab_generative:
    st.markdown("### Inverse Generative Concrete Mix Prescription")
    gen_col1, gen_col2 = st.columns([1, 2], gap="large")

    with gen_col1:
        st.markdown("#### 1. Optimization Criteria")
        sel_scm = st.selectbox("Design SCM Pozzolan:", ["RHA", "POFA", "SCBA"], key="gen_scm")
        target_mode = st.radio("Specification Route:", ["Eurocode 2 Strength Class", "Custom Characteristic Capacity (f_ck)", "Discrete Performance Tier"])

        df_pareto = pareto_results.get(sel_scm, pd.DataFrame())

        if df_pareto.empty:
            st.error(f"Pareto frontier for {sel_scm} not loaded.")
            selected_solution = None
        else:
            if target_mode == "Eurocode 2 Strength Class":
                target_ec = st.selectbox("Required Eurocode Class:", ["C16/20", "C20/25", "C25/30"], index=1)
                f_ck_req = float(target_ec.split("/")[0].replace("C", ""))
                candidates = df_pareto[0.80 * df_pareto["Predicted_28d_Strength"] >= (f_ck_req + 8.0)]
                selected_solution = candidates.sort_values("GWP_kgCO2e_m3", ascending=True).iloc[0] if not candidates.empty else df_pareto.sort_values("Predicted_28d_Strength", ascending=False).iloc[0]
            elif target_mode == "Custom Characteristic Capacity (f_ck)":
                custom_fck = st.slider("Target Characteristic Cylinder Strength (MPa):", 12.0, 28.0, 20.0, 0.5)
                candidates = df_pareto[0.80 * df_pareto["Predicted_28d_Strength"] >= (custom_fck + 8.0)]
                selected_solution = candidates.sort_values("GWP_kgCO2e_m3", ascending=True).iloc[0] if not candidates.empty else df_pareto.sort_values("Predicted_28d_Strength", ascending=False).iloc[0]
            else:
                tier_pick = st.selectbox("Standardized Decision Tier:", ["Tier 1: High-Strength", "Tier 2: Balanced Eco-Structural", "Non-structural: plain concrete only"])
                tier_sub = df_decision_matrix[(df_decision_matrix["SCM_Type"] == sel_scm) & (df_decision_matrix["Design_Tier"].str.contains(tier_pick.split(":")[0]))]
                selected_solution = tier_sub.iloc[0] if not tier_sub.empty else df_pareto.iloc[len(df_pareto)//2]

    with gen_col2:
        if selected_solution is not None:
            st.markdown("#### 2. Prescribed Batch Formulation (kg per m³ concrete)")
            cem_q = selected_solution.get("Cement_Content", selected_solution.get("Cement_kg", 0.0))
            scm_q = selected_solution.get("SCM_Mass", selected_solution.get("SCM_kg", 0.0))
            wat_q = selected_solution.get("Water_Content", selected_solution.get("Water_kg", 0.0))
            fa_q = selected_solution.get("Fine_Aggregate", selected_solution.get("Fine_Agg_kg", 0.0))
            ca_q = selected_solution.get("Coarse_Aggregate", selected_solution.get("Coarse_Agg_kg", 0.0))
            sp_q = selected_solution.get("Superplasticizer_Mass", selected_solution.get("SP_kg", 0.0))
            pred_s = selected_solution.get("Predicted_28d_Strength", selected_solution.get("Strength_MPa", 0.0))
            rep_r = selected_solution.get("SCM_Replacement_Pct", selected_solution.get("Actual_%SCM", 0.0))
            wb_r = selected_solution.get("Water_to_Binder_Ratio", selected_solution.get("w_b_ratio", 0.0))
            gwp_val = selected_solution.get("GWP_kgCO2e_m3", 0.0)
            ci_val = selected_solution.get("Eco_Efficiency_Ci", 0.0)

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("CEM I Cement", f"{cem_q:.1f} kg")
            b2.metric(f"{sel_scm} Pozzolan", f"{scm_q:.1f} kg", f"{rep_r:.1f}% repl.")
            b3.metric("Water", f"{wat_q:.1f} kg", f"w/b: {wb_r:.3f}")
            b4.metric("Superplasticizer", f"{sp_q:.2f} kg")

            b5, b6, b7, b8 = st.columns(4)
            b5.metric("Fine Aggregate", f"{fa_q:.1f} kg")
            b6.metric("Coarse Aggregate", f"{ca_q:.1f} kg")
            b7.metric("Fresh Density", f"{(cem_q+scm_q+wat_q+fa_q+ca_q+sp_q):.1f} kg/m³")
            fck_c, fck_cube_val, code_c = map_eurocode_capacity(pred_s)
            b8.metric("EC2 Class", code_c)

            st.markdown("#### 3. Sustainability & Code Conformance Card")
            card1, card2, card3 = st.columns(3)
            card1.info(f"Structural Capacity:\n\n• Predicted cube f_cu: {pred_s:.2f} MPa\n• Mean f_cm,cyl (0.80 f_cu): {0.80*pred_s:.2f} MPa\n• Char. f_ck,cyl (f_cm − 8): {fck_c:.2f} MPa")
            card2.info(f"Embodied Carbon:\n\n• Mix GWP: {gwp_val:.1f} kg CO2e/m³\n• Saving: {saving_text(gwp_val, code_c)}" + ("\n\n⚠ Plain / non-structural concrete only" if code_c in ("C12/15", "C16/20") else ""))
            card3.info(f"Eco-Efficiency:\n\n• Mix Ci: {ci_val:.2f} kg CO2e/(m³·MPa)\n• Clinker floor (author-imposed): ≥ 180 kg/m³, met")

            st.markdown("#### 4. Location Along the Continuous NSGA-II Pareto Frontier")
            if not df_pareto.empty:
                fig_par, ax_par = plt.subplots(figsize=(9, 3.8), dpi=200)
                df_sorted = df_pareto.sort_values("SCM_Replacement_Pct").reset_index(drop=True)
                ci_smooth = df_sorted["Eco_Efficiency_Ci"].rolling(window=5, min_periods=1, center=True).median()
                ax_par.plot(df_sorted["SCM_Replacement_Pct"], ci_smooth, color="#0055b3", lw=2.2, label=f"Pareto Frontier ({sel_scm})")
                ax_par.axhline(OPC_REF["Ci"], color="#d90429", linestyle=":", lw=1.2, label=f"CEM I C20/25 reference ({OPC_REF['Ci']:.2f})")
                ax_par.scatter([rep_r], [ci_val], color="#d90429", s=130, marker="*", zorder=5, label="Selected Formulation")
                ax_par.set_xlabel("Agro-Waste Replacement Percentage, %SCM (%)", fontweight="bold", fontsize=9)
                ax_par.set_ylabel("Binder Eco-Efficiency (Ci)", fontweight="bold", fontsize=9)
                ax_par.grid(True, linestyle="--", alpha=0.3)
                ax_par.legend(frameon=True, fontsize=7.5, loc="upper right")
                st.pyplot(fig_par)
                plt.close(fig_par)

# --- TAB 3: DECISION MATRIX ---
with tab_matrix:
    st.markdown("### Standardised Generative Decision Matrix")
    if not df_decision_matrix.empty:
        df_display = df_decision_matrix.copy()
        df_display = df_display[~((df_display["SCM_Type"] == "CEM I") & (df_display["Target_Class"] == "C16/20"))]
        if "f_ck_cyl_MPa" not in df_display.columns:
            df_display["f_ck_cyl_MPa"] = df_display["Strength_MPa"].apply(lambda x: max(0.0, x - 8.0))
            df_display["EC2_Class"] = df_display["Strength_MPa"].apply(lambda x: map_eurocode_capacity(x)[2])

        col_order = ["SCM_Type", "Design_Tier", "Actual_%SCM", "Strength_MPa", "f_ck_cyl_MPa", "EC2_Class", "Cement_kg", "SCM_kg", "Water_kg", "w_b_ratio", "GWP_kgCO2e_m3", "Eco_Efficiency_Ci", "CO2_Abatement_Pct"]
        st.dataframe(df_display[col_order], width="stretch")
        st.caption("Savings are against the lowest-carbon CEM I mix of the same class. C16/20 has no like-for-like CEM I reference. C25/30 savings are not reported: no CEM I mix on the front reaches C25/30, so there is no like-for-like reference, and every front flattens near the model's predictive ceiling.")
        
        csv_buffer = io.StringIO()
        df_display[col_order].to_csv(csv_buffer, index=False)
        st.download_button("📥 Export Decision Matrix as CSV", data=csv_buffer.getvalue(), file_name="concrete_decision_matrix.csv", mime="text/csv")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #6c757d; font-size: 0.82rem;'>Sustainable Concrete Engine | Master of Science in Civil Engineering | Class mapping after BS EN 1992-1-1; carbon accounting after EN 15804 (A1–A3). Screening tool, not a design check.</div>", unsafe_allow_html=True)
