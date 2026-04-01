import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import PostgresChatMessageHistory

from src.api.v1.tools.vector_tool import vector_search_tool
from src.api.v1.tools.fts_tool import fts_search_tool
from src.api.v1.tools.hybrid_tool import hybrid_search_tool

tools = [vector_search_tool, fts_search_tool, hybrid_search_tool]

# Initialize LLM
# llm = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", temperature=0)
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional Banking & Wealth Advisor.

    ### SEARCH LOGIC (STRICT HIERARCHY):
    1. **FTS-FIRST (Precision Mode)**: If the user query contains a specific technical term, acronym (e.g., ULIP, SIP, EMI, FD, Debt Fund), or unique ID, you MUST call `fts_search_tool` first using ONLY that keyword.
    2. **VECTOR-SECOND (Semantic Mode)**: Use `vector_search_tool` only for broad goals or "feelings" (e.g., "I'm worried about the future") AFTER checking for specific terms.
    3. **QUERY ATOMICITY**: Never search using full sentences. Use 1-3 word queries ONLY (e.g., instead of "how to beat inflation," search "inflation risk").
    4. **STOP CONDITION**: If a tool returns specific data, STOP and synthesize the answer. Do not perform redundant searches.

    ### CORE OPERATIONAL RULE:
    - Respond ONLY to the most recent input: "{input}".
    - Use chat history ONLY for context; do not let it pollute your search keywords.
    ### CONFIDENCE SCORE RULE:
     - confidence": "A numeric confidence score between 0 and 1 (e.g., 0.87)",
    ### OUTPUT FORMAT (STRICT):
    Return ONLY a JSON object. No prose outside the JSON.
    {{
        "answer": "Direct answer. Bold **key terms/figures**. If no data found in tools, state 'I could not find this in the internal knowledge base.' End with *Source: [Tool Name]*,\\\\n*Page no : [page]*, Citations : [source]*, Confidence : [confidence]* ",
        "policy_citations": "Code or 'N/A'",
        "page": "page" or "N/A",
        "document_name": "N/A"
    }}
    """),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def get_session_history(session_id: str):
    conn_string = os.getenv("PG_CONNECTION_STRING","").replace("postgresql+psycopg","postgresql")
    return PostgresChatMessageHistory(
        connection_string=conn_string,
        session_id=session_id,
        table_name="chat_history"
    )

agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

import json

def ask_agent(query: str, session_id: str) -> dict:
    response = agent_with_memory.invoke(
        {"input": query},
        config={
            "configurable": {"session_id": session_id},
            "tags":["Retail-Banking-agent", "Capstone-Project-1"],
            "metadata":{
                "user_id":"Team-2",
                "Feature":"Personal advisor",
                "env":"dev"
            },
            "run_name":"Retail-Banking-Agent"
            }
    )
    
    output = response.get("output")
    raw_text = ""

    # 1. Extract the raw text string from the model output
    if isinstance(output, list):
        for block in output:
            if isinstance(block, dict) and "text" in block:
                raw_text = block["text"]
                break
            elif isinstance(block, str):
                raw_text = block
                break
    else:
        raw_text = str(output)

    # 2. Parse the text into a dictionary for the FastAPI Router
    try:
        # We strip any markdown code blocks if the model includes them
        clean_json = raw_text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        # Fallback if the model fails to generate valid JSON
        print(f"JSON Parsing Error: {e}")
        return {
            "answer": raw_text,
            "policy_citations": "N/A",
            "page_no": "N/A",
            "document_name": "N/A"
        }
    

    git config --global user.email "karthick631746@gmail.com"
    git config --global user.name "karthick631746"