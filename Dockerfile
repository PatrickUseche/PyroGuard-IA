FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser
COPY app ./app
COPY models ./models
COPY data ./data
COPY scripts ./scripts
# El servicio no necesita privilegios de root; solo lee el modelo.
USER appuser
EXPOSE 8001
# /health es público y devuelve "degraded" (200) si el modelo no cargó; aquí solo se comprueba que el proceso responde.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8001'), timeout=4)"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
