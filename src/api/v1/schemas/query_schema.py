from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class RetrievedChunk(BaseModel):
    content: str
    file_name: str
    page_no: str
    confidence: Optional[float] = None
    tool_used: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    policy_citations : str
    page_no: str
    document_name : str
    retrieved_chunks: List[RetrievedChunk] = []

    
class CustomerProfile(BaseModel):
    customer_id: Optional[str] = None
    name: Optional[str] = None
    preferences: Optional[List[str]] = None
    past_interactions: Optional[List[str]] = None
    metadata: Optional[Dict] = None


class QueryRequest(BaseModel):
    query: str
    session_id: str
    customer_profile: Optional[CustomerProfile] = None 


def format_customer_profile(profile: CustomerProfile) -> str:
    if not profile:
        return ""

    parts = []

    if profile.name:
        parts.append(f"Name: {profile.name}")

    if profile.customer_id:
        parts.append(f"Customer ID: {profile.customer_id}")

    if profile.preferences:
        parts.append(f"Preferences: {', '.join(profile.preferences)}")

    if profile.past_interactions:
        parts.append(f"Past Interactions: {', '.join(profile.past_interactions)}")

    if profile.metadata:
        parts.append(f"Metadata: {profile.metadata}")

    return "\n".join(parts)
