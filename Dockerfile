FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate synthetic data and build the SQLite DB at image build time so the
# container is immediately queryable on first run.
RUN python data/generate_synthetic_data.py --n_customers 1500 --n_days 180 --out_dir data \
    && python -m backend.db.db_utils

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
