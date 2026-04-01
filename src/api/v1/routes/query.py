from fastapi import APIRouter
from src.api.v1.agents.agent_service import ask_agent
from src.api.v1.schemas.query_schema import QueryRequest, QueryResponse, CustomerProfile
from src.api.v1.schemas.query_schema import format_customer_profile

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):

    profile_context = format_customer_profile(request.customer_profile)

    if profile_context:
        enriched_query = f"""
Customer Context:
{profile_context}

User Question:
{request.query}
"""
    else:
        enriched_query = request.query

    # ✅ Step 3: call agent with enriched query
    data = ask_agent(enriched_query, request.session_id)

    return QueryResponse(
        query=request.query,
        answer=data.get("answer"),
        policy_citations=data.get("policy_citations", "N/A"),
        page_no=str(data.get("page_no", "N/A")),
        document_name=data.get("document_name", "N/A"),
    )