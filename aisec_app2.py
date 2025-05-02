import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from fpdf import FPDF
from io import BytesIO
import base64

# Set page config
st.set_page_config(page_title="URL Classification Dashboard", layout="wide", page_icon="🌐")

# ==================== Sidebar ====================
with st.sidebar:
    # Sidebar title
    st.title("🔧 Settings")
    
    # Team information
    st.header("👨‍💻 Team Members")
    team_members = [
        {"name": "Omar Ibrahim Ahmed", "id": "2206209", "dep.": "Cybersecurity"},
        {"name": "Hadeer Amr Fawzy", "id": "22010450", "dep.": "Cybersecurity"},
        {"name": "Farida Ahmed", "id": "2206160", "dep.": "Cybersecurity"},
        {"name": "Muhammed Salah", "id": "22010448", "dep.": "Cybersecurity"},
        {"name": "Marwan Gaber Ramdan", "id": "2206167", "dep.": "Cybersecurity"},
        {"name": "Youssef Tamer Muhammed Ali", "id": "2206208", "dep.": "Cybersecurity"},
        {"name": "Ahmed Saber Ahmed", "id": "2203185", "dep.": "AI"}
    ]
    
    for member in team_members:
        with st.expander(f"👤 {member['name']}"):
            st.write(f"**ID:** {member['id']}")
            st.write(f"**Department:** {member['dep.']}")
    
    # Divider
    st.markdown("---")
    
    # Data Input Section
    st.header("📂 Data Input")
    uploaded_file = st.file_uploader(
        "Upload your data file (CSV or Excel)", 
        type=["csv", "xlsx", "xls"],
        help="Upload your dataset in CSV or Excel format"
    )
    
    use_sample_data = st.checkbox(
        "Use sample data", 
        value=False, 
        help="Use built-in sample dataset"
    )

# ==================== Main page ====================
# Title and description
st.title("🌐 URL Classification Dashboard")
st.write("""
Analyze and classify URLs as malicious or benign. Upload your dataset or use sample data to explore visualizations 
and train machine learning models.
""")

# Function to load sample data
@st.cache_data
def load_sample_data():
    data = {
        'domain': [f'example{i}.com' for i in range(100)],
        'ranking': np.random.uniform(1000, 10000000, 100),
        'mld_res': np.random.choice([0, 1], 100),
        'mld.ps_res': np.random.choice([0, 1], 100),
        'card_rem': np.random.randint(1, 20, 100),
        'ratio_Rrem': np.random.uniform(10, 500, 100),
        'ratio_Arem': np.random.uniform(10, 500, 100),
        'jaccard_RR': np.random.uniform(0, 1, 100),
        'jaccard_RA': np.random.uniform(0, 1, 100),
        'jaccard_AR': np.random.uniform(0, 1, 100),
        'jaccard_AA': np.random.uniform(0, 1, 100),
        'jaccard_ARrd': np.random.uniform(0, 1, 100),
        'jaccard_ARrem': np.random.uniform(0, 1, 100),
        'label': np.random.choice([0, 1], 100)
    }
    return pd.DataFrame(data)

# Load data
if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    st.session_state.data = data
elif use_sample_data:
    data = load_sample_data()
    st.session_state.data = data
else:
    st.warning("⚠️ Please upload a data file or select 'Use sample data' from the sidebar.")
    st.stop()
    
# Data preprocessing
cols_to_float = [
    "ranking", "mld_res", "mld.ps_res", "card_rem",
    "ratio_Rrem", "ratio_Arem", "jaccard_RR",
    "jaccard_RA", "jaccard_AR", "jaccard_AA",
    "jaccard_ARrd", "jaccard_ARrem"
]

for col in cols_to_float:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

if "label" in data.columns:
    data["label"] = pd.to_numeric(data["label"], errors="coerce").fillna(0).astype(int)

data.drop_duplicates(inplace=True)
data.dropna(inplace=True)


# Data visualization section
st.header("📊 Data Visualization")
viz_type = st.selectbox(
    "Select visualization type",
    ["Distribution Plot", "Box Plot", "Scatter Plot", "Correlation Heatmap", "Count Plot"]
)

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = data.select_dtypes(include=['object']).columns.tolist()

if viz_type in ["Distribution Plot", "Box Plot"]:
    col_to_plot = st.selectbox("Select column to plot", numeric_cols)
    fig, ax = plt.subplots(figsize=(10, 6))
    if viz_type == "Distribution Plot":
        sns.histplot(data[col_to_plot], kde=True, ax=ax)
    else:
        sns.boxplot(x=data[col_to_plot], ax=ax)
    st.pyplot(fig)

elif viz_type == "Scatter Plot":
    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("X-axis", numeric_cols)
    with col2:
        y_axis = st.selectbox("Y-axis", numeric_cols)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x=x_axis, y=y_axis, ax=ax)
    st.pyplot(fig)

elif viz_type == "Correlation Heatmap":
    selected_cols = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:5])
    if len(selected_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data[selected_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

elif viz_type == "Count Plot":
    col_to_count = st.selectbox("Select column", cat_cols + ["label"] if "label" in data.columns else cat_cols)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=data, x=col_to_count, ax=ax)
    st.pyplot(fig)

# Machine Learning Section
st.header("🤖 Machine Learning Models")

features = st.multiselect(
    "Select features",
    options=data.columns.drop(['domain', 'label'] if 'label' in data.columns else ['domain']),
    default=['ranking', 'mld_res', 'card_rem', 'ratio_Rrem']
)

target = 'label' if 'label' in data.columns else None
if target is None:
    st.warning("No 'label' column found.")
    st.stop()

X = data[features]
y = data[target]

test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
random_state = st.slider("Random state", 0, 100, 42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models dictionary in session state
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

# Train individual model
model_type = st.selectbox("Select a model", ["Logistic Regression", "Random Forest", "SVM"])

if st.button("🚀 Train Model"):
    with st.spinner("Training model..."):
        if model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=random_state)
        elif model_type == "Random Forest":
            model = RandomForestClassifier(random_state=random_state)
        else:
            model = SVC(probability=True, random_state=random_state)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Store model with its metrics
        st.session_state.trained_models[model_type] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'scaler': scaler,
            'features': features
        }
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
            st.text(classification_report(y_test, y_pred))
        with col2:
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
            st.pyplot(fig)
        
        if model_type == "Random Forest":
            fig, ax = plt.subplots()
            importances = model.feature_importances_
            sns.barplot(x=features, y=importances)
            plt.xticks(rotation=45)
            st.pyplot(fig)

# Compare all models
if st.checkbox("Compare all models"):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(probability=True)
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Classification Report": classification_report(y_test, y_pred),
            "Confusion Matrix": confusion_matrix(y_test, y_pred)
        })
    
    st.table(pd.DataFrame(results)[["Model", "Accuracy"]])
    fig, ax = plt.subplots()
    sns.barplot(x="Accuracy", y="Model", data=pd.DataFrame(results))
    st.pyplot(fig)

# URL Prediction Section
st.header("🕵️‍♂️ Predict Suspicious or Safe? URLs")

if st.session_state.trained_models:
    url = st.text_input("Enter URL (e.g., example.com)", "")
    
    if st.button("Predict"):
        if not url:
            st.warning("Please enter a URL")
        else:
            # Example feature extraction - replace with your actual logic
            input_values = {feature: np.random.uniform(0, 1) for feature in features}
            input_df = pd.DataFrame([input_values])
            
            st.subheader(f"Prediction for: {url}")
            cols = st.columns(len(st.session_state.trained_models))
            
            for i, (name, model_data) in enumerate(st.session_state.trained_models.items()):
                with cols[i]:
                    scaled_input = model_data['scaler'].transform(input_df)
                    prediction = model_data['model'].predict(scaled_input)[0]
                    proba = model_data['model'].predict_proba(scaled_input)[0][1] if hasattr(model_data['model'], "predict_proba") else None
                    
                    st.markdown(f"**{name}**")
                    st.error("Suspicious") if prediction == 1 else st.success("Safe")
                    if proba:
                        st.write(f"Confidence: {proba:.1%}")
                        fig, ax = plt.subplots(figsize=(4, 0.5))
                        ax.barh([''], [proba], color='red' if prediction == 1 else 'green')
                        ax.set_xlim(0, 1)
                        st.pyplot(fig)

# PDF Report Generation
def generate_pdf():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "URL Classification Report", ln=1, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align='C')
    pdf.ln(10)
    
    # Data Summary
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "1. Data Summary", ln=1)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 5, f"Dataset contains {len(data)} URLs with {len(features)} features.")
    
    # Model Results
    if st.session_state.trained_models:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, "2. Model Performance", ln=1)
        
        for name, model_data in st.session_state.trained_models.items():
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, name, ln=1)
            pdf.set_font("Arial", size=10)
            pdf.multi_cell(0, 5, f"Accuracy: {model_data['accuracy']:.2%}")
            pdf.multi_cell(0, 5, "Classification Report:")
            pdf.multi_cell(0, 5, model_data['report'])
    
    return pdf.output(dest='S').encode('latin1')

# Report Download
st.header("📄 Generate Report")
if st.button("Generate PDF Report"):
    if not st.session_state.trained_models:
        st.warning("Please train at least one model first")
    else:
        pdf_bytes = generate_pdf()
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="url_report.pdf">Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)