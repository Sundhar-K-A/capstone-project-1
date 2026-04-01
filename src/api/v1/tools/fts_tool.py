import json
import os
import psycopg
from psycopg.rows import dict_row
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Setup raw connection for direct SQL execution
_raw_conn = os.getenv("PG_CONNECTION_STRING", "").replace("postgresql+psycopg", "postgresql")

@tool
def fts_search_tool(query: str) -> str:
    """
    Use this tool for looking up technical terms, abbreviations, or specific 
    financial products like 'ULIP', 'SIP', or 'EMI'.
    """

    sql = """
        SELECT
            e.document AS content,
            e.cmetadata AS metadata,
            ts_rank(to_tsvector('english', e.document), websearch_to_tsquery('english', %(query)s)) AS fts_rank
        FROM langchain_pg_embedding e
        WHERE to_tsvector('english', e.document) @@ websearch_to_tsquery('english', %(query)s)
        ORDER BY fts_rank DESC
        LIMIT 5;
    """
    
    try:
        with psycopg.connect(_raw_conn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"query": query})
                rows = cur.fetchall()

        if not rows:
            return "No exact matches found in the internal retail banking documents."

        max_rank = max(float(r["fts_rank"]) for r in rows) or 1.0
        results = [
            {
                "content":  row["content"],
                "metadata": row["metadata"],
                "fts_rank": round(float(row["fts_rank"]) / max_rank, 4),
            }
            for row in rows
        ]
        return json.dumps(results)
        
    except Exception as e:
        return f"Error executing FTS search: {str(e)}"
