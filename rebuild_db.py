# rebuild_db.py
import os
from backend.db import db_utils

if os.path.exists("aml.db"):
    os.remove("aml.db")
db_utils.init_db()