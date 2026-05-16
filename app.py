import sqlite3
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Segmentation System",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
    """
    <style>

    /* Main Background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Titles */
    h1 {
        color: #2563eb;
        font-weight: 700;
    }

    h2, h3 {
        color: #0f172a;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white;
    }

    /* Metrics Cards */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
    }

    /* Success Box */
    .stSuccess {
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# DATABASE CONNECTION
# =====================================================

conn = sqlite3.connect(
    "customers.db",
    check_same_thread=False
)

cursor = conn.cursor()

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("mall_customers.csv")

# =====================================================
# FEATURE SELECTION
# =====================================================

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# =====================================================
# SCALING
# =====================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# =====================================================
# MODEL TRAINING
# =====================================================

kmeans = KMeans(
    n_clusters=5,
    random_state=42
)

df['Cluster'] = kmeans.fit_predict(X_scaled)

# =====================================================
# CLUSTER LABELS
# =====================================================

cluster_names = {
    0: "Premium Customer",
    1: "Budget Customer",
    2: "Regular Shopper",
    3: "High Potential Customer",
    4: "Inactive Customer"
}

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("ML Customer App")

page = st.sidebar.selectbox(
    "Choose Page",
    ["Dashboard", "Prediction", "History", "Analytics"]
)

# =====================================================
# DASHBOARD PAGE
# =====================================================

if page == "Dashboard":

    st.title("Customer Segmentation Dashboard")

    st.markdown(
        """
        <div style='background: linear-gradient(90deg, #2563eb, #7c3aed);
                    padding: 25px;
                    border-radius: 15px;
                    color: white;
                    margin-bottom: 20px;'>

            <h2>AI-Powered Customer Analytics</h2>

            <p>
            Analyze customer behavior using Machine Learning and K-Means Clustering.
            </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    # Metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Customers",
        len(df)
    )

    col2.metric(
        "Average Income",
        round(df['Annual Income (k$)'].mean(), 2)
    )

    col3.metric(
        "Average Spending Score",
        round(df['Spending Score (1-100)'].mean(), 2)
    )

    st.subheader("Dataset Preview")

    st.dataframe(df.head())

    # Download Button
    csv = df.to_csv(index=False)

    st.download_button(
        label="Download Dataset",
        data=csv,
        file_name='customers.csv',
        mime='text/csv'
    )

# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "Prediction":

    st.title("Predict Customer Segment")

    income = st.number_input(
        "Enter Annual Income",
        min_value=0
    )

    score = st.number_input(
        "Enter Spending Score",
        min_value=0,
        max_value=100
    )

    if st.button("Predict Customer Segment"):

        # Scale Input
        input_data = scaler.transform(
            [[income, score]]
        )

        # Predict
        prediction = kmeans.predict(input_data)

        cluster_id = int(prediction[0])

        cluster_label = cluster_names.get(
            cluster_id,
            "Unknown"
        )

        # Save to Database
        cursor.execute(
            """
            INSERT INTO customers
            (income, spending_score, cluster)
            VALUES (?, ?, ?)
            """,
            (
                income,
                score,
                cluster_label
            )
        )

        conn.commit()

        st.success(
            "Prediction saved to database"
        )

        st.subheader("Prediction Result")

        st.success(
            f"Customer Type: {cluster_label}"
        )

        # Colored Info Cards
        if cluster_label == "Premium Customer":

            st.markdown(
                """
                <div style='padding:15px;
                            background:#dcfce7;
                            border-radius:10px;'>

                Premium users are high-value customers.

                </div>
                """,
                unsafe_allow_html=True
            )

        elif cluster_label == "Budget Customer":

            st.markdown(
                """
                <div style='padding:15px;
                            background:#fee2e2;
                            border-radius:10px;'>

                Budget customers prefer discounts.

                </div>
                """,
                unsafe_allow_html=True
            )

        elif cluster_label == "Regular Shopper":

            st.markdown(
                """
                <div style='padding:15px;
                            background:#dbeafe;
                            border-radius:10px;'>

                Regular shoppers provide stable sales.

                </div>
                """,
                unsafe_allow_html=True
            )

        elif cluster_label == "High Potential Customer":

            st.markdown(
                """
                <div style='padding:15px;
                            background:#fef9c3;
                            border-radius:10px;'>

                High potential customers can become premium buyers.

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div style='padding:15px;
                            background:#e2e8f0;
                            border-radius:10px;'>

                Inactive customers need engagement.

                </div>
                """,
                unsafe_allow_html=True
            )

# =====================================================
# HISTORY PAGE
# =====================================================

elif page == "History":

    st.title("Prediction History")

    history_data = pd.read_sql_query(
        "SELECT * FROM customers",
        conn
    )

    st.dataframe(history_data)

# =====================================================
# ANALYTICS PAGE
# =====================================================

elif page == "Analytics":

    st.title("Customer Analytics")

    # Scatter Plot
    fig, ax = plt.subplots(figsize=(10,6))

    scatter = ax.scatter(
        df['Annual Income (k$)'],
        df['Spending Score (1-100)'],
        c=df['Cluster'],
        cmap='rainbow'
    )

    ax.set_xlabel("Annual Income")

    ax.set_ylabel("Spending Score")

    ax.set_title("Customer Segments")

    st.pyplot(fig)

    # Pie Chart
    st.subheader("Customer Distribution")

    pie_fig = px.pie(
        df,
        names='Cluster',
        title='Customer Segments'
    )

    st.plotly_chart(pie_fig)

    # Business Insights
    with st.expander("Business Insights"):

        st.write(
            "Premium customers generate high revenue."
        )

        st.write(
            "Budget customers respond well to discounts."
        )

        st.write(
            "Regular shoppers provide stable sales."
        )

        st.write(
            "Inactive customers need re-engagement."
        )