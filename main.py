import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from openai import AzureOpenAI

load_dotenv(".env")

# --- Config from environment ---
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
QDRANT_URL = os.getenv("QDRANT_URL")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT")
AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "3072"))
COLLECTION_NAME = "legal_kg"

# --- Initialize clients ---
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
qdrant_client = QdrantClient(url=QDRANT_URL)
openai_client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)

# --- Pydantic models ---
class GraphQueryRequest(BaseModel):
    node_ids: List[int]

class VectorSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QARequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    max_attempts: Optional[int] = 2


app = FastAPI(
    title="Legal Knowledge Graph QA API",
    description="API for querying a legal knowledge graph using Neo4j, Qdrant, and Azure OpenAI.",
    version="1.0"
)

# --- Utility functions ---

def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Azure OpenAI."""
    response = openai_client.embeddings.create(
        input=text,
        model=AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT
    )
    return response.data[0].embedding

def retrieve_from_qdrant(query: str, top_k: int = 5):
    """Run vector search on Qdrant."""
    query_embedding = get_embedding(query)
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k
    )
    return results

def fetch_graph_context(node_ids: List[int]):
    """Fetch neighborhood subgraph from Neo4j for given node IDs."""
    with neo4j_driver.session() as session:
        # Outgoing edges (one hop)
        outgoing = session.run(
            """
            MATCH (n)-[r]->(m)
            WHERE id(n) IN $ids
            RETURN id(n) as src_id, type(r) as rel_type, id(m) as tgt_id,
                   labels(n) as src_labels, properties(n) as src_props,
                   labels(m) as tgt_labels, properties(m) as tgt_props
            """,
            ids=[int(x) for x in node_ids if x is not None]
        )

        # Incoming paths recursively (ancestors)
        incoming = session.run(
            """
            MATCH path = (ancestor)-[r*1..]->(n)
            WHERE id(n) IN $ids
            WITH nodes(path) AS ns, relationships(path) AS rs
            UNWIND range(0, size(rs)-1) AS idx
            RETURN
                id(ns[idx]) AS src_id,
                type(rs[idx]) AS rel_type,
                id(ns[idx + 1]) AS tgt_id,
                labels(ns[idx]) AS src_labels,
                properties(ns[idx]) AS src_props,
                labels(ns[idx + 1]) AS tgt_labels,
                properties(ns[idx + 1]) AS tgt_props
            """,
            ids=[int(x) for x in node_ids if x is not None]
        )

        edges = []

        for rec in outgoing:
            edges.append({
                "src_id": rec["src_id"],
                "tgt_id": rec["tgt_id"],
                "rel_type": rec["rel_type"],
                "src_labels": rec["src_labels"],
                "tgt_labels": rec["tgt_labels"],
                "src_props": rec["src_props"],
                "tgt_props": rec["tgt_props"],
            })

        for rec in incoming:
            edges.append({
                "src_id": rec["src_id"],
                "tgt_id": rec["tgt_id"],
                "rel_type": rec["rel_type"],
                "src_labels": rec["src_labels"],
                "tgt_labels": rec["tgt_labels"],
                "src_props": rec["src_props"],
                "tgt_props": rec["tgt_props"],
            })

        return edges

def is_relevant_query(query: str) -> bool:
    """Classify if query is legal related (Yes/No) using LLM."""
    prompt = f"""
    You are a classifier. Decide if this question is about legal matters and laws:

    Question: "{query}"

    Answer only "Yes" or "No".
    """
    resp = openai_client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Classify the question topic."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3,
        temperature=0,
    )
    answer = resp.choices[0].message.content.strip().lower()
    return answer == "yes"

def validate_answer(query: str, answer: str) -> bool:
    """Validate if the generated answer sufficiently answers the query."""
    prompt = f"""
    Given the user question: "{query}"
    And the generated answer: "{answer}"

    Does the answer fully address the user's question? Answer Yes or No.
    """
    resp = openai_client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Validate if answer is sufficient."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3,
        temperature=0,
    )
    return resp.choices[0].message.content.strip().lower() == "yes"

def generate_answer(query: str, top_nodes: List[dict], subgraph: List[dict]) -> str:
    """Generate final answer using LLM given context nodes and subgraph edges."""
    context = ""
    for node in top_nodes:
        context += f"\nNode: [{','.join(node.get('labels', []))}] " + "; ".join(f"{k}={v}" for k,v in node.items() if k not in ['labels', 'node_id'])

    rel_context = ""
    for edge in subgraph:
        src_name = edge['src_props'].get('name', '') or edge['src_props'].get('title', '')
        tgt_name = edge['tgt_props'].get('name', '') or edge['tgt_props'].get('title', '')
        rel_context += f"\nEdge: {src_name} -[{edge['rel_type']}]-> {tgt_name}"

    prompt = f"""You are an expert legal assistant.
You have access to the following knowledge graph:
{context}
{rel_context}

User Query: {query}
Please answer using only the information provided above.
Always provide the specific law, article, section, and subsection your answer is based on, as references.
"""
    response = openai_client.chat.completions.create(
        model=AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "Provide an accurate, well-grounded answer."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def run_qa_pipeline(question: str, top_k: int = 3, max_attempts: int = 2) -> str:
    """Run the full QA loop: classify, vector search, graph fetch, LLM generation + validation."""
    if not is_relevant_query(question):
        raise HTTPException(status_code=400, detail="Question is not legal-related.")

    attempt = 0
    current_top_k = top_k
    answer = None

    while attempt < max_attempts:
        qdrant_results = retrieve_from_qdrant(question, top_k=current_top_k)
        node_ids = [res.payload.get('node_id') for res in qdrant_results]
        top_nodes = [res.payload for res in qdrant_results]

        subgraph = fetch_graph_context(node_ids)
        answer = generate_answer(question, top_nodes, subgraph)

        if validate_answer(question, answer):
            return answer

        attempt += 1
        current_top_k += 2

    # If no good answer found
    return "Sorry, I couldn't find a satisfactory answer."

# --- API Endpoints ---

@app.post("/query/graph")
async def api_query_graph(req: GraphQueryRequest):
    edges = fetch_graph_context(req.node_ids)
    return {"edges": edges}

@app.post("/query/vector_search")
async def api_vector_search(req: VectorSearchRequest):
    results = retrieve_from_qdrant(req.query, top_k=req.top_k)
    # Return only payload to keep response concise
    return {"results": [r.payload for r in results]}

@app.post("/query/answer")
async def api_answer(req: QARequest):
    answer = run_qa_pipeline(req.question, top_k=req.top_k, max_attempts=req.max_attempts)
    return {"answer": answer}
