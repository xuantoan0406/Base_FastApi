from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def read_users():
    return [{"user_id": 1, "name": "user1"}, {"user_id": 2, "name": "user2"}]
