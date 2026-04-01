
import json
from langchain_core.tools import tool
from src.api.v1.tools.vector_tool import vector_search_tool
from src.api.v1.tools.fts_tool import fts_search_tool

@tool
def hybrid_search_tool(query: str) -> str:
    """
    The PRIMARY search tool. Use this for almost all complex questions 
    that combine a policy name with a specific question. It is the most 
    reliable tool for accurate answers.
    """
    # 1. Get raw results from other tools
    # Since tools return JSON strings, we parse them back to lists
    vector_results = json.loads(vector_search_tool.invoke(query))
    fts_results = json.loads(fts_search_tool.invoke(query))

    # 2. Reciprocal Rank Fusion (RRF) Logic
    rrf_scores = {}
    chunk_map = {}

    for rank, doc in enumerate(vector_results):
        key = doc["content"][:120] # Using a prefix as a unique key for merging
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        chunk_map[key] = doc

    for rank, doc in enumerate(fts_results):
        key = doc["content"][:120]
        rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (60 + rank + 1)
        chunk_map[key] = doc

    # 3. Sort by score and return top results
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    final_results = [chunk_map[key] for key, _ in ranked[:5]]
    
    return json.dumps(final_results)