from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_users():
    response = client.get("/users/")
    assert response.status_code == 200
    assert response.json() == [{"user_id": 1, "name": "user1"}, {"user_id": 2, "name": "user2"}]
