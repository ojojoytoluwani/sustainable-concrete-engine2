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
    "Cement": 0.8300,
    "RHA": 0.0500,
    "POFA": 0.0350,
    "SCBA": 0.0350,
    "Water": 0.0002,
    "Fine_Agg": 0.0050,
    "Coarse_Agg": 0.0080,
    "SP": 1.2500
}

OPC_REF = {
    "Cement": 400.0, "Fine_Agg": 720.0, "Coarse_Agg": 1100.0,
    "Water": 160.0, "SP": 2.0, "fc": 40.0,
    "GWP": 346.93, "Ci": 8.67
}

DEFAULT_OXIDES = {
    "RHA": {"SiO2": 90.90, "Al2O3": 0.80, "Fe2O3": 0.40, "CaO": 0.90, "K2O": 2.20, "LOI": 3.80},
    "POFA": {"SiO2": 58.20, "Al2O3": 5.40, "Fe2O3": 6.80, "CaO": 5.10, "K2O": 6.20, "LOI": 10.30},
    "SCBA": {"SiO2": 72.40, "Al2O3": 6.80, "Fe2O3": 5.20, "CaO": 4.10, "K2O": 3.40, "LOI": 5.10}
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

# 4. Eurocode 2 & BS 8500 Auditing Functions
def evaluate_structural_compliance(cement, scm, water, fine, coarse, sp, scm_type):
    total_binder = cement + scm
    wb_ratio = water / total_binder if total_binder > 0 else 999.0
    fresh_density = cement + scm + water + fine + coarse + sp
    rep_pct = (scm / total_binder) * 100.0 if total_binder > 0 else 0.0
    
    violations = []
    warnings = []
    
    if cement < 180.0:
        violations.append(
            f"Clinker Starvation: Portland cement ({cement:.1f} kg/m³) is below the 180 kg/m³ "
            "clinker clamping boundary. Critical risk of rebar depassivation under BS 8500."
        )
        
    if total_binder < 260.0:
        violations.append(
            f"Binder Deficit: Total binder ({total_binder:.1f} kg/m³) is below the BS EN 206 "
            "minimum structural durability floor (260 kg/m³)."
        )
    elif total_binder > 550.0:
        warnings.append(
            f"Thermal Cracking Risk: Total binder ({total_binder:.1f} kg/m³) exceeds 550 kg/m³ limit."
        )
        
    if wb_ratio < 0.28:
        warnings.append(f"Low w/b Ratio ({wb_ratio:.3f}): Risk of incomplete hydration and compaction.")
    elif wb_ratio > 0.60:
        violations.append(f"Excess Porosity: w/b ({wb_ratio:.3f}) exceeds structural maximum limit (0.60).")
        
    if fresh_density < 2200.0 or fresh_density > 2550.0:
        warnings.append(
            f"Fresh Density ({fresh_density:.1f} kg/m³) deviates from standard normal-weight concrete (2200–2550 kg/m³)."
        )

    exposures = []
    if wb_ratio <= 0.65 and total_binder >= 260: exposures.append("XC1 (Dry Interior)")
    if wb_ratio <= 0.60 and total_binder >= 280: exposures.append("XC2 (Foundation / Wet Soil)")
    if wb_ratio <= 0.55 and total_binder >= 300: exposures.append("XC3/XC4 (External Facades)")
    if wb_ratio <= 0.45 and total_binder >= 340: exposures.append("XD1/XS1 (Chlorides / Seawater Spray)")

    if rep_pct <= 5.0:
        cement_type = "CEM I (Ordinary Portland Cement)"
    elif rep_pct <= 20.0:
        cement_type = f"CEM II/A-P ({scm_type} Pozzolanic 6–20%)"
    elif rep_pct <= 35.0:
        cement_type = f"CEM II/B-P ({scm_type} Pozzolanic 21–35%)"
    else:
        cement_type = f"CEM IV/B ({scm_type} High-Volume Pozzolanic 36–50%)"

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
    f_ck_cyl = max(0.0, f_cm - 8.0)
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

tab_forward, tab_generative, tab_matrix = st.tabs([
    "🔬 Forward Prediction & SHAP Audit",
    "⚡ Inverse Generative Designer (NSGA-II)",
    "📋 Eurocode Decision Matrix & LCA Scorecard"
])

# --- TAB 1: FORWARD PREDICTION ---
with tab_forward:
    st.markdown("### Structural Capacity Estimation & Mechanistic Attribution")

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

        water_val = st.number_input("Batching Water:", min_value=100.0, max_value=260.0, value=160.0, step=5.0)
        fine_val = st.number_input("Natural River Sand:", min_value=500.0, max_value=1000.0, value=750.0, step=10.0)
        coarse_val = st.number_input("Crushed Granite (Coarse):", min_value=800.0, max_value=1350.0, value=1050.0, step=10.0)
        sp_val = st.number_input("Superplasticizer (SP):", min_value=0.0, max_value=15.0, value=2.5, step=0.5)

    with col_chem:
        st.markdown("#### 2. Curing, Milling & Chemistry")
        age_val = st.slider("Curing Age (Days):", min_value=3, max_value=90, value=28, step=1)
        control_str_val = st.number_input(
            "Reference Lab 28d Control Strength (MPa):", 
            min_value=20.0, max_value=70.0, value=40.0, step=1.0,
            help="28-day compressive strength of 100% Portland cement control mix from the testing facility."
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
        other_ox_val = max(0.0, 100.0 - (sio2_val + al2o3_val + fe2o3_val + cao_val + k2o_val + loi_val))

    input_dict = {
        "Cement_Content": cement_val,
        "SCM_Mass": scm_val,
        "Water_Content": water_val,
        "Fine_Aggregate": fine_val,
        "Coarse_Aggregate": coarse_val,
        "Superplasticizer_Mass": sp_val,
        "Curing_Age": age_val,
        "Control_Strength_28d": control_str_val,
        "Grinding_Intensity_enc": grind_val,
        "SCM_SiO2": sio2_val,
        "SCM_Al2O3": al2o3_val,
        "SCM_Fe2O3": fe2o3_val,
        "SCM_CaO": cao_val,
        "SCM_K2O": k2o_val,
        "SCM_LOI": loi_val,
        "SCM_Other_Oxides": other_ox_val,
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
    eco_index_ci = audit_res["GWP"] / pred_fcm if pred_fcm > 0 else 999.0

    with col_pred:
        st.markdown("#### 3. Eurocode 2 Designations")
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size:0.9rem; color:#666;">EUROCODE 2 DESIGNATION</span><br>
            <span style="font-size:1.8rem; font-weight:700; color:#0055b3;">{ec2_class}</span><br>
            <span style="font-size:0.85rem; color:#333;">Permissible in: {', '.join(audit_res['exposures'][:2]) if audit_res['exposures'] else 'Non-Structural'}</span>
        </div>
        """, unsafe_allow_html=True)
        
        s1, s2, s3 = st.columns(3)
        s1.metric("Mean f_cm", f"{pred_fcm:.1f} MPa")
        s2.metric("Char. f_ck,cyl", f"{f_ck_cyl:.1f} MPa")
        s3.metric("Char. f_ck,cube", f"{f_ck_cube:.1f} MPa")

        st.markdown("#### 4. Environmental Accounting (EN 15804)")
        e1, e2, e3 = st.columns(3)
        e1.metric("Embodied Carbon", f"{audit_res['GWP']:.1f} kg CO₂e/m³")
        e2.metric("Carbon Abated", f"{audit_res['Abatement_Pct']:.1f}%")
        e3.metric("Eco-Efficiency (Ci)", f"{eco_index_ci:.2f}")

        st.markdown("#### 5. BS EN 206 Durability Conformance")
        if audit_res["is_compliant"]:
            st.markdown(f'<div class="audit-pass">✓ PASSED: Compliant with BS EN 206. ({audit_res["cement_type"]})</div>', unsafe_allow_html=True)
        else:
            fail_text = "<br>• ".join(audit_res["violations"])
            st.markdown(f'<div class="audit-fail">✗ VIOLATION:<br>• {fail_text}</div>', unsafe_allow_html=True)

        if audit_res["warnings"]:
            for w in audit_res["warnings"]: st.warning(w)

    st.markdown("---")
    st.markdown("### Mechanistic Attribution: TreeSHAP Decomposition")
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
                target_ec = st.selectbox("Required Eurocode Class:", ["C20/25", "C25/30", "C30/37", "C32/40"], index=2)
                f_ck_req = float(target_ec.split("/")[0].replace("C", ""))
                candidates = df_pareto[df_pareto["Predicted_28d_Strength"] >= (f_ck_req + 8.0)]
                selected_solution = candidates.sort_values("GWP_kgCO2e_m3", ascending=True).iloc[0] if not candidates.empty else df_pareto.sort_values("Predicted_28d_Strength", ascending=False).iloc[0]
            elif target_mode == "Custom Characteristic Capacity (f_ck)":
                custom_fck = st.slider("Target Characteristic Cylinder Strength (MPa):", 20.0, 36.0, 28.0, 0.5)
                candidates = df_pareto[df_pareto["Predicted_28d_Strength"] >= (custom_fck + 8.0)]
                selected_solution = candidates.sort_values("GWP_kgCO2e_m3", ascending=True).iloc[0] if not candidates.empty else df_pareto.sort_values("Predicted_28d_Strength", ascending=False).iloc[0]
            else:
                tier_pick = st.selectbox("Standardized Decision Tier:", ["Tier 1: High-Strength", "Tier 2: Balanced Eco-Structural", "Tier 3: High-Volume Eco"])
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
            card1.info(f"Structural Capacity:\n\n• Mean f_cm: {pred_s:.2f} MPa\n• Char. f_ck,cyl: {fck_c:.2f} MPa\n• Char. f_ck,cube: {fck_cube_val:.2f} MPa")
            card2.info(f"Embodied Carbon:\n\n• Mix GWP: {gwp_val:.1f} kg CO2e/m³\n• Savings: {(((346.93-gwp_val)/346.93)*100):.1f}% vs OPC")
            card3.info(f"Eco-Efficiency:\n\n• Mix Ci: {ci_val:.2f} kg CO2e/(m³·MPa)\n• Clinker Floor: ≥ 180 kg/m³ (Met)")

            st.markdown("#### 4. Location Along the Continuous NSGA-II Pareto Frontier")
            if not df_pareto.empty:
                fig_par, ax_par = plt.subplots(figsize=(9, 3.8), dpi=200)
                df_sorted = df_pareto.sort_values("SCM_Replacement_Pct").reset_index(drop=True)
                ci_smooth = df_sorted["Eco_Efficiency_Ci"].rolling(window=5, min_periods=1, center=True).median()
                ax_par.plot(df_sorted["SCM_Replacement_Pct"], ci_smooth, color="#0055b3", lw=2.2, label=f"Pareto Frontier ({sel_scm})")
                ax_par.axhline(OPC_REF["Ci"], color="#d90429", linestyle=":", lw=1.2, label="OPC Baseline (8.67)")
                ax_par.scatter([rep_r], [ci_val], color="#d90429", s=130, marker="*", zorder=5, label="Selected Formulation")
                ax_par.set_xlabel("Agro-Waste Replacement Percentage, %SCM (%)", fontweight="bold", fontsize=9)
                ax_par.set_ylabel("Binder Eco-Efficiency (Ci)", fontweight="bold", fontsize=9)
                ax_par.grid(True, linestyle="--", alpha=0.3)
                ax_par.legend(frameon=True, fontsize=7.5, loc="upper right")
                st.pyplot(fig_par)
                plt.close(fig_par)

# --- TAB 3: DECISION MATRIX ---
with tab_matrix:
    st.markdown("### Standardized Generative Decision Matrix (Table 8.1)")
    if not df_decision_matrix.empty:
        df_display = df_decision_matrix.copy()
        if "f_ck_cyl_MPa" not in df_display.columns:
            df_display["f_ck_cyl_MPa"] = df_display["Strength_MPa"].apply(lambda x: max(0.0, x - 8.0))
            df_display["EC2_Class"] = df_display["Strength_MPa"].apply(lambda x: map_eurocode_capacity(x)[2])

        col_order = ["SCM_Type", "Design_Tier", "Actual_%SCM", "Strength_MPa", "f_ck_cyl_MPa", "EC2_Class", "Cement_kg", "SCM_kg", "Water_kg", "w_b_ratio", "GWP_kgCO2e_m3", "Eco_Efficiency_Ci", "CO2_Abatement_Pct"]
        st.dataframe(df_display[col_order], use_container_width=True)
        
        csv_buffer = io.StringIO()
        df_display[col_order].to_csv(csv_buffer, index=False)
        st.download_button("📥 Export Decision Matrix as CSV", data=csv_buffer.getvalue(), file_name="concrete_decision_matrix.csv", mime="text/csv")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #6c757d; font-size: 0.82rem;'>Sustainable Concrete Engine | Master of Science in Civil Engineering | BS EN 1992-1-1 & EN 15804 Compliance</div>", unsafe_allow_html=True)
