# init_db.py
from app import app
from models import db

with app.app_context():
    db.create_all()
    print("✅ 数据库表已创建成功。")
