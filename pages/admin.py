import streamlit as st
import requests

st.set_page_config(page_title="Admin Panel", layout="wide")

st.title("🛠 Admin Panel")

st.subheader("Upload Knowledge Base")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    try:
        res = requests.post(
            "http://localhost:8000/api/v1/admin/upload",
            files={"file": uploaded_file}
        )

        if res.status_code == 200:
            st.success("File uploaded & processed")
        else:
            st.error(f" Upload failed: {res.text}")

    except Exception as e:
        st.error(f"Error: {str(e)}")