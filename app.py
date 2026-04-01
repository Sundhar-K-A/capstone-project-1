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

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Select Customer")
customer_names = [c["name"] for c in CUSTOMERS]
selected_name = st.sidebar.selectbox("Choose Customer", customer_names)
selected_customer = next(c for c in CUSTOMERS if c["name"] == selected_name)
customer_profile = selected_customer["profile"]

with st.sidebar.expander("View Profile"):
    st.json(customer_profile)

st.sidebar.divider()

if st.sidebar.button("New Session", use_container_width=True):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()

st.sidebar.caption(f"Session: `{st.session_state.session_id[:16]}…`")

# ── Render existing chat ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            doc      = msg.get("document_name", "")
            page_no  = msg.get("page_no", "")
            citation = msg.get("policy_citations", "")
            chunks   = msg.get("retrieved_chunks", [])
            meta     = " · ".join(filter(None, [doc, f"Page {page_no}" if page_no and page_no != "N/A" else None]))

            if (citation and citation != "N/A") or chunks:
                label = f"📎 Source & Chunks — {meta}" if meta else " Source & Chunks"
                with st.expander(label):
                    if citation and citation != "N/A":
                        st.markdown(f"**Citation:** {citation}")
                        if chunks:
                            st.divider()
                    if chunks:
                        st.json(chunks)

# ── Chat input ─────────────────────────────────────────────────────────────────
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
                "age":    customer_profile.get("age"),
                "income": customer_profile.get("income"),
                "risk":   customer_profile.get("risk"),
            },
        },
    }

    answer = "No response"
    data   = {}

    try:
        with st.spinner("Searching..."):
            res  = requests.post("http://localhost:8000/api/v1/query", json=payload, timeout=60)
            res.raise_for_status()
            data = res.json()
    except requests.exceptions.ConnectionError:
        answer = "Cannot reach the API. Is FastAPI running on port 8000?"
    except Exception as e:
        answer = f"Error: {str(e)}"

    answer           = data.get("answer", answer)
    policy_citations = data.get("policy_citations", "")
    page_no          = data.get("page_no", "")
    document_name    = data.get("document_name", "")
    retrieved_chunks = data.get("retrieved_chunks", [])

    with st.chat_message("assistant"):
        st.markdown(answer)

        # Combined source + chunks dropdown
        meta = " · ".join(filter(None, [document_name, f"Page {page_no}" if page_no and page_no != "N/A" else None]))
        if (policy_citations and policy_citations != "N/A") or retrieved_chunks:
            label = f"📎 Source & Chunks — {meta}" if meta else " Source & Chunks"
            with st.expander(label):
                if policy_citations and policy_citations != "N/A":
                    st.markdown(f"**Citation:** {policy_citations}")
                    if retrieved_chunks:
                        st.divider()
                if retrieved_chunks:
                    st.json(retrieved_chunks)

    st.session_state.messages.append({
        "role":              "assistant",
        "content":           answer,
        "policy_citations":  policy_citations,
        "page_no":           page_no,
        "document_name":     document_name,
        "retrieved_chunks":  retrieved_chunks,
    })
