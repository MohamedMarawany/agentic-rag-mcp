# GraphDB integration (Neo4j example)
from neo4j import GraphDatabase

class GraphDB:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def add_node(self, label, properties):
        with self.driver.session() as session:
            session.run(f"CREATE (n:{label} $props)", props=properties)
