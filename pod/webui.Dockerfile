# Observability webui. Host network only — never attached to the pod's
# networks (the pod must not see its own dashboard).
FROM python:3.12-slim
RUN pip install --no-cache-dir "fastapi>=0.115" "uvicorn>=0.30"
WORKDIR /app
COPY src/sts2_webui/ sts2_webui/
ENV STS2_DB=/data/sts2.sqlite3 POD_HOME=/podhome
EXPOSE 8310
CMD ["uvicorn", "sts2_webui.app:app", "--host", "0.0.0.0", "--port", "8310"]
