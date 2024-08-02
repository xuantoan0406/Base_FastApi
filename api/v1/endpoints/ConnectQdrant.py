from qdrant_client import QdrantClient, models
from core.config import settings


class ConnectQdrant:
    def __init__(self, host, port):
        self.client = QdrantClient(host=host, port=port)

    def create_collection(self, name_collection):
        self.client.create_collection(
            collection_name=name_collection,  # Name of the collection
            vectors_config=models.VectorParams(
                size=1024,  # Size of the vectors to be stored
                distance=models.Distance.COSINE  # Distance metric for vector similarity
            )
        )
        print("success")

    def disconnect(self):
        self.client.delete_collection("my_collection")
        print("success")

    def insert(self, collection_name, id_data, data):
        self.client.upsert(
            collection_name,
            points=[
                {
                    "id": id_data,
                    "vector": data
                }
            ]
        )
        return "insert success"
