import streamlit as st
import requests
from customer_data import CUSTOMERS
import uuid

st.set_page_config(page_title="Financial Assistant", layout="wide")

st.title("Financial Assistant")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.header("Select Customer")

customer_names = [c["name"] for c in CUSTOMERS]

selected_name = st.sidebar.selectbox("Choose Customer", customer_names)

selected_customer = next(c for c in CUSTOMERS if c["name"] == selected_name)

customer_profile = selected_customer["profile"]

with st.sidebar.expander("View Profile"):
    st.json(customer_profile)

# Toggle
# show_data = st.sidebar.toggle("Show Retrieved Data")
show_data = True
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask your question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
    "query": prompt,
    "session_id": st.session_state.session_id,
    "customer_profile": {
        "customer_id": selected_customer["customer_id"],
        "name": selected_customer["name"],
        "preferences": customer_profile.get("preferences"),
        "past_interactions": customer_profile.get("past_interactions"),
        "metadata": {
            "age": customer_profile.get("age"),
            "income": customer_profile.get("income"),
            "risk": customer_profile.get("risk")
        }
    }
    }

    try:
        res = requests.post("http://localhost:8000/api/v1/query", json=payload)

        response_json = res.json()

        answer = response_json.get("answer", "No response")
        retrieved_docs = response_json.get("retrieved_docs", [])

    except Exception as e:
        answer = f"Error: {str(e)}"
        retrieved_docs = []

    with st.chat_message("assistant"):
        st.markdown(answer)

        if show_data and retrieved_docs:
            st.markdown("###Retrieved Docs")
            for i, doc in enumerate(retrieved_docs):
                st.write(f"Doc {i+1}: {doc}")

    st.session_state.messages.append({"role": "assistant", "content": answer})