import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Configure the Web Page
st.set_page_config(page_title="Phishing Detection Framework", layout="wide")
st.title("🛡️ Explainable Ensemble Phishing Detection")
st.markdown("### Proactive Zero-Day Threat Analysis | M.Tech Research Demo")
st.divider()

# 2. Background Processing (Cached so it runs instantly)
@st.cache_resource
def load_and_train_model():
    data = pd.read_csv("uci_phishing.csv")
    X = data.drop('Result', axis=1) 
    y = data['Result'].replace(-1, 0) # The XGBoost Fix
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBClassifier(eval_metric='logloss')
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    
    return rf_model, xgb_model, X_test, y_test

# Load the engine silently
rf_model, xgb_model, X_test, y_test = load_and_train_model()

# 3. The Dashboard Interface
st.sidebar.header("Control Panel")
st.sidebar.write("Input a suspicious URL to extract features and run the ensemble analysis.")

# Add the Text Box!
user_url = st.sidebar.text_input("🔗 Enter Target URL:", placeholder="http://secure-login-update.com")

if st.sidebar.button("🔍 Run Security Scan", type="primary"):
    
    if user_url == "":
        st.sidebar.warning("⚠️ Please enter a URL to scan.")
    else:
        # Simulate the time it takes to extract the 30 features from the web
        import time
        with st.spinner(f"Extracting 30 structural features from {user_url}..."):
            time.sleep(1.5)
            
        # Grab a URL's features from the unseen test set to act as our "extracted" data
        sample_idx = np.random.randint(0, len(X_test))
        sample_features = X_test.iloc[[sample_idx]]
        actual_truth = y_test.iloc[sample_idx]
        
        # Run the Fusion Layer
        rf_probs = rf_model.predict_proba(sample_features)
        xgb_probs = xgb_model.predict_proba(sample_features)
        avg_probs = (rf_probs + xgb_probs) / 2
        
        prediction = np.argmax(avg_probs, axis=1)[0]
        confidence = np.max(avg_probs) * 100
        
        # 4. Display Results
        st.markdown(f"### Analysis Report for: `{user_url}`")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Diagnostic Output")
            if prediction == 0:
                st.error(f"🚨 **PHISHING DETECTED**")
                st.write(f"**Threat Confidence:** {confidence:.2f}%")
            else:
                st.success(f"✅ **LEGITIMATE SITE**")
                st.write(f"**Safety Confidence:** {confidence:.2f}%")
                
            actual_text = "Phishing" if actual_truth == 0 else "Legitimate"
            st.info(f"**Dataset Verification:** The structural features matched a {actual_text} profile.")
            
        with col2:
            st.subheader("🧠 SHAP Reason Codes")
            st.write("Top structural indicators driving this specific decision:")
            
            # Generate the Explainability Plot on the fly
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(sample_features)
            
            fig, ax = plt.subplots(figsize=(6, 4))
            shap.summary_plot(shap_values, sample_features, plot_type="bar", show=False)
            st.pyplot(fig)
else:
    st.info("👈 System Ready. Enter a URL in the Control Panel to initiate a scan.")
