import os
from tqdm import tqdm
from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from openai import AzureOpenAI
import uuid
from dotenv import load_dotenv

load_dotenv('.env')  # Loads variables from .env into the environment
# --- CONFIG --- #
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")

AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_MODEL_DEPLOYMENT")
AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT")
VECTOR_DIM = int(os.environ.get("VECTOR_DIM", "3072"))

COLLECTION_NAME = "legal_kg"
# --- INIT CLIENTS --- #
openai_client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION,
)
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# --- CREATE QDRANT COLLECTION IF NOT EXISTS --- #
def create_collection_qdrant(client, collection_name, vector_dim):
    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        if "Not found" in str(e):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_dim, distance=models.Distance.COSINE)
            )
            print(f"Created collection '{collection_name}'.")
        else:
            raise

create_collection_qdrant(qdrant_client, COLLECTION_NAME, VECTOR_DIM)

# --- EXTRACT ALL NODES AND FIELDS FROM NEO4J --- #
def get_all_nodes_and_fields(driver):
    query = """
    MATCH (n)
    RETURN labels(n) as labels, id(n) as node_id, properties(n) as props
    """
    with driver.session() as session:
        records = session.run(query)
        nodes = []
        for record in records:
            node = {
                "labels": record["labels"],     # e.g. ['Law'], ['Article'], etc
                "node_id": record["node_id"],   # Neo4j internal node id
                **record["props"],              # all node properties flattened
            }
            nodes.append(node)
        return nodes

# --- EMBEDDING UTILITY --- #
def get_embedding(text):
    response = openai_client.embeddings.create(
        input=text,
        model=AZURE_OPENAI_EMBED_MODEL_DEPLOYMENT
    )
    return response.data[0].embedding

# --- GENERATE TEXT FOR EMBEDDING --- #
def serialize_node(node):
    s = f"[{','.join(node['labels'])}] " + "; ".join([f"{k}={v}" for k,v in node.items() if k not in ['labels', 'node_id']])
    return s

# --- INGEST EMBEDDINGS TO QDRANT --- #
import uuid

def ingest_all_embeddings(nodes, batch_size=50):
    points_batch = []
    for node in tqdm(nodes, desc="Generating embeddings and uploading in batches"):
        text = serialize_node(node)
        vector = get_embedding(text)
        # Use UUID as point ID, save Neo4j node_id in payload
        points_batch.append({
            "id": str(uuid.uuid4()),      # Always a valid string UUID!
            "vector": vector,
            "payload": node               # Save everything you want here, including 'node_id'
        })
        if len(points_batch) >= batch_size:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points_batch
            )
            points_batch.clear()
    if points_batch:
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points_batch
        )
# --- MAIN PIPELINE --- #
if __name__ == "__main__":
    print("Extracting all nodes from Neo4j...")
    nodes = get_all_nodes_and_fields(neo4j_driver)
    print(f"Found {len(nodes)} nodes.")

    print("Generating and ingesting embeddings...")
    ingest_all_embeddings(nodes)
    print("Done ingesting embeddings to Qdrant.")
