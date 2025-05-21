import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from openai import AzureOpenAI

# --- Load env variables ---
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
QDRANT_URL = os.getenv("QDRANT_URL")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT")
AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "3072"))
COLLECTION_NAME = "legal_kg"  # Edit if your Qdrant collection name differs

# --- OpenAI / Azure API setup ---
# --- INIT CLIENTS --- #
openai_client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)
# --- Connect to Neo4j and Qdrant ---
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
qdrant_client = QdrantClient(url=QDRANT_URL)

# --- Embedding Function (using Azure OpenAI embedding deployment) ---
def get_embedding(text):
    resp = openai_client.Embedding.create(
        input=text,
        model=AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT
    )
    return resp['data'][0]['embedding']
def get_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model=AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT
    )
    return response.data[0].embedding
# --- Qdrant Retrieval ---
def retrieve_from_qdrant(query, top_k=1):
    query_embedding = get_embedding(query)
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k
    )
    # print(results)
    return results

def is_relevant_query(query):
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
def validate_answer(query, answer):
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

# --- Neo4j: fetch context subgraph for given node_ids ---
def fetch_graph_context(node_ids):
    with neo4j_driver.session() as session:
        # Get immediate outgoing edges (one hop)
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

        # Get all incoming paths recursively (ancestors)
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

        # Collect outgoing edges
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

        # Collect incoming ancestor edges (recursive)
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

        # print(edges)
        return edges


# --- LLM Generation using Azure OpenAI ---
def generate_answer(query, top_nodes, subgraph):
    # Format the context for LLM
    context = ""
    for node in top_nodes:
        context += f"\nNode: [{','.join(node.get('labels', []))}] " + "; ".join(f"{k}={v}" for k, v in node.items() if k not in ['labels', 'node_id'])

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

# --- MAIN INTERACTION LOOP ---
if __name__ == "__main__":
    print("Hybrid Legal QA Agent. Type your question (or 'exit'):")
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            print("Please ask a question.")
            continue
        if not is_relevant_query(query):
            print("Sorry, I can only answer legal-related questions. Please ask about laws or legal topics.")
            continue

        max_attempts = 2
        top_k = 3
        answer = None

        for attempt in range(max_attempts):
            print(f"Retrieving top {top_k} nodes from Qdrant...")
            qdrant_results = retrieve_from_qdrant(query, top_k=top_k)
            node_ids = [res.payload.get('node_id') for res in qdrant_results]
            top_nodes = [res.payload for res in qdrant_results]
            print("Fetching neighborhood from Neo4j...")
            subgraph = fetch_graph_context(node_ids)
            print("Generating answer using LLM...")
            answer = generate_answer(query, top_nodes, subgraph)
            print("\n--- Generated answer ---")
            print(answer)

            if validate_answer(query, answer):
                print("\nAnswer validated as sufficient.")
                break
            else:
                print("\nAnswer insufficient, retrying with more context...")
                top_k += 2  # Increase context window

        if not answer:
            print("Sorry, I couldn't find a satisfactory answer.")
