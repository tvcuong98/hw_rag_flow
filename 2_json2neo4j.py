from neo4j import GraphDatabase
import json
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv('.env')  # Loads variables from .env into the environment
# --- CONFIG --- #
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USER = os.environ.get("NEO4J_USER")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

# Neo4j Write Transactions
def create_law(tx, law_id, title, agency):
    tx.run(
        "MERGE (l:Law {law_id: $law_id}) "
        "SET l.title = $title, l.agency = $agency",
        law_id=law_id, title=title, agency=agency
    )

def create_article(tx, law_id, article_name):
    tx.run(
        "MATCH (l:Law {law_id: $law_id}) "
        "CREATE (a:Article {name: $name})<-[:HAS_ARTICLE]-(l)",
        law_id=law_id, name=article_name
    )

def create_title(tx, article_name, title_name):
    tx.run(
        "MATCH (a:Article {name: $article_name}) "
        "CREATE (t:Title {name: $name})<-[:HAS_TITLE]-(a)",
        article_name=article_name, name=title_name
    )

def create_paragraph(tx, title_name, para_name):
    tx.run(
        "MATCH (t:Title {name: $title_name}) "
        "CREATE (p:Paragraph {name: $name})<-[:HAS_PARAGRAPH]-(t)",
        title_name=title_name, name=para_name
    )

def create_subsection(tx, para_name, sub_name, sub_content):
    tx.run(
        "MATCH (p:Paragraph {name: $para_name}) "
        "CREATE (s:Subsection {name: $name, content: $content})<-[:HAS_SUBSECTION]-(p)",
        para_name=para_name, name=sub_name, content=sub_content
    )

def create_agency(tx, law_id, agency_name):
    tx.run(
        "MATCH (l:Law {law_id: $law_id}) "
        "MERGE (a:Agency {name: $agency_name}) "
        "MERGE (l)-[:ENFORCED_BY]->(a)",
        law_id=law_id, agency_name=agency_name
    )

# Process JSON Files to Neo4j
def process_json(filepath, driver):
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    law_id = data["law_id"]
    title = data["title"]
    agency = data["agency"]

    with driver.session() as session:
        session.execute_write(create_law, law_id, title, agency)
        for article in data.get("structure", []):
            session.execute_write(create_article, law_id, article["name"])
            for title_obj in article.get("titles", []):
                session.execute_write(create_title, article["name"], title_obj["name"])
                for para in title_obj.get("paragraphs", []):
                    session.execute_write(create_paragraph, title_obj["name"], para["name"])
                    for sub in para.get("subsections", []):
                        session.execute_write(create_subsection, para["name"], sub["name"], sub["content"])
        # Process agency enforcement
        session.execute_write(create_agency, law_id, agency)

def main(input_folder):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]
    for fname in tqdm(json_files, desc="Seeding Neo4j"):
        process_json(os.path.join(input_folder, fname), driver)
    driver.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", required=True, help="Folder containing input JSON files")
    args = parser.parse_args()
    main(args.input_folder)
