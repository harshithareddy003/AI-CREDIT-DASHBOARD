import os
import random
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from faker import Faker

try:
    import google.generativeai as genai
except ImportError:
    genai = None


gemini_available = genai is not None
if gemini_available:
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    except Exception:
        gemini_available = False


st.set_page_config(page_title="Credit Dashboard + Gemini Bot", layout="wide")
st.title("Credit Dashboard: Generate, Analyze, Ask")

if not os.path.exists("regulations.txt"):
    with open("regulations.txt", "w") as f:
        f.write("This is a placeholder for financial regulations like Basel III. Add your context here.")

tab1, tab2 = st.tabs(["Data Generator", "Dashboard + Chatbot"])

with tab1:
    st.subheader("Synthetic Credit Data Generator")
    fake = Faker()
    num_rows = st.number_input("Number of rows to generate:", min_value=10, max_value=10000, value=100)

    if st.button("Generate Data"):
        with st.spinner("Generating synthetic data..."):
            data = []
            for _ in range(num_rows):
                data.append(
                    {
                        "CustomerID": fake.unique.bothify(text="CUST-#####"),
                        "PD": round(random.uniform(0.01, 0.3), 4),
                        "LGD": round(random.uniform(0.2, 0.8), 4),
                        "EAD": round(random.uniform(50000, 10000000), 2),
                        "CreditRating": random.choice(["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]),
                        "Sector": random.choice(["Retail", "Manufacturing", "Technology", "Banking", "Insurance"]),
                        "Region": random.choice(["North", "South", "East", "West"]),
                        "Date": fake.date_between(start_date="-2y", end_date="today"),
                    }
                )
            df = pd.DataFrame(data)
            st.session_state.generated_df = df
            st.success("Data generated successfully.")
            st.dataframe(df)
            st.download_button(
                label="Download as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="synthetic_credit_data.csv",
                mime="text/csv",
            )

with tab2:
    st.header("Interactive Dashboard & Chatbot")
    df = None
    uploaded_file = st.file_uploader(
        "Upload your own CSV (optional, will use generated data if available)", type="csv"
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_df = df
    elif "generated_df" in st.session_state:
        df = st.session_state.generated_df
    else:
        st.warning("Please generate data in Tab 1 or upload a CSV file to begin.")
        st.stop()

    try:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    except Exception as e:
        st.error("Error converting 'Date' column to datetime: {}".format(e))
        st.stop()

    st.subheader("Credit Risk Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Average Probability of Default (PD)", "{:.2%}".format(df["PD"].mean()))
    col2.metric("Average Loss Given Default (LGD)", "{:.2%}".format(df["LGD"].mean()))
    col3.metric("Total Exposure at Default (EAD)", "${:.2f}M".format(df["EAD"].sum() / 1_000_000))

    c1, c2 = st.columns((6, 4))
    with c1:
        st.markdown("#### EAD Over Time")
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        timeline = df.set_index("Date").resample("ME")["EAD"].sum().reset_index()
        sns.lineplot(data=timeline, x="Date", y="EAD", ax=ax4, marker="o", color="#3498db")
        ax4.set_title("Total EAD per Month")
        st.pyplot(fig4)

    with c2:
        st.markdown("#### Exposure by Sector")
        sector_ead = df.groupby("Sector")["EAD"].sum().sort_values(ascending=False)
        st.bar_chart(sector_ead, color="#2ecc71")

    st.header("Ask Your Credit Analysis Assistant")

    if not gemini_available:
        st.info(
            "Gemini chat is unavailable in this environment because `google-generativeai` "
            "or the API key is missing. The dashboard will still work."
        )

    @st.cache_resource
    def create_retriever(_df_hash, txt_path="regulations.txt"):
        docs = []
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                docs.append(f.read())

        df_for_docs = st.session_state.get("uploaded_df", st.session_state.get("generated_df"))
        rows = df_for_docs.astype(str).apply(lambda x: " | ".join(x.values), axis=1).tolist()
        docs.extend(rows)
        return docs

    def retrieve_relevant_context(query, docs, top_k=5):
        query_terms = set(re.findall(r"\w+", query.lower()))
        scored_docs = []

        for doc in docs:
            doc_lower = doc.lower()
            doc_terms = set(re.findall(r"\w+", doc_lower))
            overlap_score = len(query_terms & doc_terms)
            substring_score = sum(1 for term in query_terms if term and term in doc_lower)
            total_score = overlap_score * 2 + substring_score
            if total_score > 0:
                scored_docs.append((total_score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        if not scored_docs:
            return docs[:top_k]
        return [doc for _, doc in scored_docs[:top_k]]

    df_hash = pd.util.hash_pandas_object(df).sum()
    retriever_docs = create_retriever(df_hash)

    def generate_credit_prompt(query, context):
        return """
You are a helpful and insightful AI assistant integrated into a Streamlit dashboard for credit risk analysis.

Your purpose is to help users understand their credit portfolio data. The application you are part of visualizes key metrics (PD, LGD, EAD) and allows users to ask questions about the underlying data.

When answering, follow these rules:
1. For questions about specific data points, trends, customers, sectors, or regulations, you MUST use the provided CONTEXT below. Ground your answer in this data.
2. For general questions about the purpose or advantages of this application, answer based on your role. Explain how visualizing data and asking direct questions can help a risk analyst.

CONTEXT:
{}

QUESTION:
{}

ANSWER:
""".format(context, query)

    def get_gemini_answer(query):
        if not gemini_available:
            return "Gemini is not available in this Python environment, so chat responses are disabled right now."

        try:
            docs = retrieve_relevant_context(query, retriever_docs)
            context = "\n---\n".join(docs[:5])
            prompt = generate_credit_prompt(query, context)
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            st.error("An error occurred while generating the response: {}".format(e))
            return "Sorry, I ran into a problem. Please check the logs or try rephrasing your question."

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask about credit risk, trends, or regulations...")
    if query:
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("Analyzing..."):
            answer = get_gemini_answer(query)
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
