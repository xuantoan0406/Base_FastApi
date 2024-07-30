from sqlalchemy.orm import Session
from models import item
from schemas import item as item_schema


def get_item(db: Session, item_id: int):
    return db.query(item.Item).filter(item.Item.id == item_id).first()

def get_items(db: Session, skip: int = 0, limit: int = 10):
    return db.query(item.Item).offset(skip).limit(limit).all()

def create_item(db: Session, item: item_schema.ItemCreate):
    db_item = item.Item(name=item.name, description=item.description)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
