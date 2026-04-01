from __future__ import annotations

import json
import os
from typing import Any
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

from src.api.v1.tools.vector_tool import vector_search_tool
from src.api.v1.tools.fts_tool import fts_search_tool
from src.api.v1.tools.hybrid_tool import hybrid_search_tool

load_dotenv()

# ---------------------------------------------------------------------------
# 1. LLM
# ---------------------------------------------------------------------------
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

# ---------------------------------------------------------------------------
# 2. System Prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an expert Banking & Wealth Advisor.

You have access to three retrieval tools:
vector_search_tool  → best for natural language / conceptual questions
fts_search_tool     → best for codes, IDs, abbreviations, exact keywords
hybrid_search_tool  → best for short or ambiguous queries

Rules:
1. Choose exactly one tool and call it with the original user query.
2. Use ONLY the returned document chunks to answer.
3. Synthesise a clear, concise answer.
4. Always return your final response as a JSON object:
{
    "answer": "your answer here",
    "policy_citations": "citation text or N/A",
    "page": "page number from chunk metadata or N/A",
    "document_name": "document name from chunk metadata or N/A"
}
5. For 'page': use metadata['page'] from the most relevant chunk. If missing, use N/A.
6. For 'document_name': use metadata['source'] or metadata['file_name'] from the most relevant chunk.
Do not add anything outside the JSON.
"""

# ---------------------------------------------------------------------------
# 3. Base agent
# ---------------------------------------------------------------------------
_base_agent = create_agent(
    model=model,
    tools=[vector_search_tool, fts_search_tool, hybrid_search_tool],
    system_prompt=_SYSTEM_PROMPT,
)

# ---------------------------------------------------------------------------
# 4. In-memory chat history store  { session_id → ChatMessageHistory }
# ---------------------------------------------------------------------------
_history_store: dict[str, ChatMessageHistory] = {}

def _get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _history_store:
        _history_store[session_id] = ChatMessageHistory()
    return _history_store[session_id]

# ---------------------------------------------------------------------------
# 5. Wrap with RunnableWithMessageHistory
# ---------------------------------------------------------------------------
agent_with_history = RunnableWithMessageHistory(
    _base_agent,
    _get_session_history,
    input_messages_key="messages",
    history_messages_key="chat_history",
)

# ---------------------------------------------------------------------------
# 6. Helper: extract raw chunks from the last tool call in the message trace
# ---------------------------------------------------------------------------
TOOL_NAMES = {"vector_search_tool", "fts_search_tool", "hybrid_search_tool"}

def _extract_chunks(messages: list) -> list:
    """Return the raw chunk dicts from the most recent tool response."""
    for msg in reversed(messages):
        if getattr(msg, "type", "") == "tool" and getattr(msg, "name", "") in TOOL_NAMES:
            try:
                return json.loads(msg.content)
            except Exception:
                pass
    return []

# ---------------------------------------------------------------------------
# 7. Public entry-point  (same signature our route already calls)
# ---------------------------------------------------------------------------
def ask_agent(query: str, session_id: str, customer_context: dict[str, Any] | None = None) -> dict:
    """Run the RAG agent and return a plain dict the route can use directly."""

    if customer_context:
        context_block = json.dumps(customer_context, indent=2, ensure_ascii=False)
        full_message = f"[Customer Context]\n{context_block}\n\n[Question]\n{query}"
    else:
        full_message = query

    result = agent_with_history.invoke(
        {"messages": [{"role": "user", "content": full_message}]},
        config={
            "configurable": {"session_id": session_id},
            "tags": ["Retail-Banking-agent", "Capstone-Project-1"],
            "metadata": {"user_id": "Team-2", "Feature": "Personal advisor", "env": "dev"},
            "run_name": "Retail-Banking-Agent",
        },
    )

    messages = result.get("messages", [])
    raw_chunks = _extract_chunks(messages)

    # Build retrieved_chunks list our schema understands
    retrieved_chunks = []
    seen: set = set()
    for doc in raw_chunks:
        if not isinstance(doc, dict):
            continue
        content = doc.get("content", "")
        if content[:100] in seen:
            continue
        seen.add(content[:100])
        meta = doc.get("metadata") or {}
        confidence = doc.get("fts_rank") or doc.get("score") or meta.get("score")
        retrieved_chunks.append({
            "content":   content,
            "file_name": meta.get("source") or meta.get("file_name") or "N/A",
            "page_no":   str(meta.get("page") or meta.get("page_number") or "N/A"),
            "confidence": round(float(confidence), 4) if confidence else None,
            "tool_used": next(
                (getattr(m, "name", "unknown") for m in reversed(messages)
                 if getattr(m, "type", "") == "tool"), "unknown"
            ),
        })

    # Parse the LLM JSON output
    final_msg = messages[-1] if messages else None
    raw_text = getattr(final_msg, "content", "") if final_msg else ""
    if isinstance(raw_text, list):
        raw_text = next((b.get("text", "") for b in raw_text if isinstance(b, dict) and "text" in b), "")

    try:
        output = json.loads(str(raw_text).replace("```json", "").replace("```", "").strip())
    except Exception:
        output = {
            "answer": str(raw_text),
            "policy_citations": "N/A",
            "page": "N/A",
            "document_name": "N/A",
        }

    output["retrieved_chunks"] = retrieved_chunks
    return output
# import os
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import PostgresChatMessageHistory

# from src.api.v1.tools.vector_tool import vector_search_tool
# from src.api.v1.tools.fts_tool import fts_search_tool
# from src.api.v1.tools.hybrid_tool import hybrid_search_tool

# tools = [vector_search_tool, fts_search_tool, hybrid_search_tool]

# # Initialize LLM
# # llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=0)
# llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)
# prompt = ChatPromptTemplate.from_messages([
#     ("system", """You are a professional Banking & Wealth Advisor.

#     ### SEARCH LOGIC (STRICT HIERARCHY):
#     1. **FTS-FIRST (Precision Mode)**: If the user query contains a specific technical term, acronym (e.g., ULIP, SIP, EMI, FD, Debt Fund), or unique ID, you MUST call `fts_search_tool` first using ONLY that keyword.
#     2. **VECTOR-SECOND (Semantic Mode)**: Use `vector_search_tool` only for broad goals or "feelings" (e.g., "I'm worried about the future") AFTER checking for specific terms.
#     3. **QUERY ATOMICITY**: Never search using full sentences. Use 1-3 word queries ONLY (e.g., instead of "how to beat inflation," search "inflation risk").
#     4. **STOP CONDITION**: If a tool returns specific data, STOP and synthesize the answer. Do not perform redundant searches.
#     ###STOP CONDITION: 
#     - If any tool returns relevant results:
#     - STOP further searching
#     - Generate final answer
#     ### CORE OPERATIONAL RULE:
#     - Respond ONLY to the most recent input: "{input}".
#     - Use chat history ONLY for context; do not let it pollute your search keywords.
#     ### CONFIDENCE SCORE RULE:
#      - confidence": "A numeric confidence score between 0 and 1 (e.g., 0.87)",
#     ### OUTPUT FORMAT (STRICT):
#     Return ONLY a JSON object. No prose outside the JSON.
#     {{
#         "answer": "Direct answer. Bold **key terms/figures**. If no data found in tools, state 'I could not find this in the internal knowledge base.' End with *Source: [Tool Name]*,\\\\n*Page no : [page]*, Citations : [source +".pdf"]*, Confidence : [confidence]* ",
#         "policy_citations": "Code or 'N/A'",
#         "page": "page" or "N/A",
#         "document_name": "N/A"
#     }}
#     """),
#     ("placeholder", "{chat_history}"),
#     ("human", "{input}"),
#     ("placeholder", "{agent_scratchpad}"),
# ])

# agent = create_tool_calling_agent(llm, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# def get_session_history(session_id: str):
#     conn_string = os.getenv("PG_CONNECTION_STRING","").replace("postgresql+psycopg","postgresql")
#     return PostgresChatMessageHistory(
#         connection_string=conn_string,
#         session_id=session_id,
#         table_name="chat_history"
#     )

# agent_with_memory = RunnableWithMessageHistory(
#     agent_executor,
#     get_session_history,
#     input_messages_key="input",
#     history_messages_key="chat_history"
# )

# import json

# def ask_agent(query: str, session_id: str) -> dict:
#     response = agent_with_memory.invoke(
#         {"input": query},
#         config={
#             "configurable": {"session_id": session_id},
#             "tags":["Retail-Banking-agent", "Capstone-Project-1"],
#             "metadata":{
#                 "user_id":"Team-2",
#                 "Feature":"Personal advisor",
#                 "env":"dev"
#             },
#             "run_name":"Retail-Banking-Agent"
#             }
#     )
    
#     output = response.get("output")
#     raw_text = ""

#     # 1. Extract the raw text string from the model output
#     if isinstance(output, list):
#         for block in output:
#             if isinstance(block, dict) and "text" in block:
#                 raw_text = block["text"]
#                 break
#             elif isinstance(block, str):
#                 raw_text = block
#                 break
#     else:
#         raw_text = str(output)

#     # 2. Parse the text into a dictionary for the FastAPI Router
#     try:
#         # We strip any markdown code blocks if the model includes them
#         clean_json = raw_text.replace("```json", "").replace("```", "").strip()
#         return json.loads(clean_json)
#     except Exception as e:
#         # Fallback if the model fails to generate valid JSON
#         print(f"JSON Parsing Error: {e}")
#         return {
#             "answer": raw_text,
#             "policy_citations": "N/A",
#             "page_no": "N/A",
#             "document_name": "N/A"
#         }
    


































































































































































































































































































































































    # ghghjjjjjjj
    

































































































































































































































































































































































































































































































# completed