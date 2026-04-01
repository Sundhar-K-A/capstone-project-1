import json
import os
from langchain_core.tools import tool
from src.core.db import get_vector_store
from dotenv import load_dotenv

load_dotenv()

@tool
def vector_search_tool(query: str) -> str:
    """
    Use this for conversational questions or when searching for topics 
    where you don't know the exact wording (e.g., 'What happens if I'm late?').
    """
    # Initialize store inside the tool to ensure fresh connection
    vector_store = get_vector_store()
    
    # Perform search
    docs = vector_store.similarity_search(query, k=5)
    
    # Format results
    results = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    return json.dumps(results)