import streamlit as st
import pandas as pd
import requests
import plotly.express as px

st.set_page_config(page_title="Production AI Framework", layout="wide", page_icon="🛡️")
BACKEND_URL = "http://127.0.0.1:8000"

st.title("🖥️ Production AI Customer Intelligence Dashboard")
st.markdown("Stateless Frontend Interface built on top of decoupled high-throughput Microservices.")

uploaded_file = st.sidebar.file_uploader("Upload Telecom Telemetry (CSV)", type=["csv"])

if uploaded_file is not None:
    # حفظ الـ Bytes الخاصة بالملف لإرسالها للـ Endpoints بأكثر من Request دون استهلاك مساحة برمجية
    file_bytes = uploaded_file.getvalue()
    files_payload = {"file": (uploaded_file.name, file_bytes, "text/csv")}
    
    with st.spinner("Processing real-time inference transaction via Backend API..."):
        try:
            response = requests.post(f"{BACKEND_URL}/predict", files=files_payload)
        except requests.exceptions.ConnectionError:
            st.error("❌ Critical Infrastructure Failure: Gateway unable to ping backend services.")
            st.stop()
            
    if response.status_code == 200:
        data = response.json()
        records = pd.DataFrame(data["records"])
        metadata = data["metadata"]
        
        # --- Metrics Rendering ---
        st.markdown(f"### 📊 System Status Layer (Engine Active Version: `v{metadata['version']}`)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Financial Exposure Risk", f"${data['total_exposure']:,.2f}")
        m2.metric("Critical Target Accounts count", data["critical_accounts_count"])
        m3.metric("Dynamic Cutoff Threshold (80th Pct)", f"{data['dynamic_high_cutoff_pct']:.2f}%")
        
        st.markdown("---")
        
        # --- Visualization Layer ---
        g1, g2 = st.columns(2)
        with g1:
            st.subheader("💰 Expected Loss by Corporate Action Group")
            fig_bar = px.bar(records, x='Risk_Level', y='Expected_Annual_Loss', color='Risk_Level',
                             color_discrete_map={'🔴 High Risk':'red', '🟡 Medium Risk':'orange', '🟢 Low Risk':'green'})
            st.plotly_chart(fig_bar, use_container_width=True)
        with g2:
            st.subheader("⚠️ Monthly Spend vs Risk Correlation Map")
            fig_scat = px.scatter(records, x='MonthlyCharges', y='Churn_Prob', color='Risk_Level',
                                  color_discrete_map={'🔴 High Risk':'red', '🟡 Medium Risk':'orange', '🟢 Low Risk':'green'})
            st.plotly_chart(fig_scat, use_container_width=True)
            
        st.markdown("---")
        
        # --- Decoupled On-Demand XAI Auditing Layer ---
        st.subheader("🔍 Decoupled XAI Audit Channel (On-Demand SHAP)")
        st.markdown("Select a specific Account ID. The frontend will trigger an isolated async request to fetch execution drivers from the API without stalling memory pipelines.")
        
        selected_client = st.selectbox("Choose Client to Audit:", records['customerID'].unique())
        
        if st.button("Trigger Isolated Audit"):
            with st.spinner("Invoking background XAI job..."):
                audit_files = {"file": (uploaded_file.name, file_bytes, "text/csv")}
                audit_res = requests.post(f"{BACKEND_URL}/audit-customer?customer_id={selected_client}", files=audit_files)
                
                if audit_res.status_code == 200:
                    drivers = pd.DataFrame(audit_res.json()["drivers"])
                    st.write(f"**Root-Cause Explanations for Client `{selected_client}` Decisions:**")
                    fig_shap = px.bar(drivers, x='shap_value', y='feature', orientation='h',
                                      color='shap_value', color_continuous_scale=px.colors.diverging.RdBu_r)
                    st.plotly_chart(fig_shap, use_container_width=True)
                else:
                    st.error(f"Audit endpoint rejected the parameter payload: {audit_res.text}")
                    
        st.markdown("---")
        st.subheader("📋 Production Ledger Ledger Instance")
        st.dataframe(records.sort_values(by='Expected_Annual_Loss', ascending=False), use_container_width=True)
    else:
        st.error(f"API Error [{response.status_code}]: {response.json().get('detail', 'Registry Error')}")
else:
    st.info("💡 Standby. Connect raw data stream to initiate operational evaluation frameworks.")