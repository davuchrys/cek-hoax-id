FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY data/sample.csv ./data/sample.csv
# Train at build time so the image is self-contained (mount a real model in production).
RUN python -m hoax_detector.train --data data/sample.csv --out models/model.joblib
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "hoax_detector.api:app", "--host", "0.0.0.0", "--port", "8000"]
