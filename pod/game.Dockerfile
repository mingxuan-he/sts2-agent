# Game service: headless sts2-cli engine + FastAPI wrapper.
# Build context is the REPO ROOT (see docker-compose.yml).
# The image contains the game DLLs — never push it to a registry.

FROM mcr.microsoft.com/dotnet/sdk:9.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Engine first (biggest layer, changes least)
COPY sts2-cli/ sts2-cli/
RUN dotnet build sts2-cli/src/Sts2Headless/Sts2Headless.csproj -c Release

# Python service
RUN python3 -m venv /venv && /venv/bin/pip install --no-cache-dir \
        "fastapi>=0.115" "uvicorn>=0.30" "httpx>=0.27"
COPY headless_env.py .
COPY src/sts2_service/ src/sts2_service/

ENV DOTNET_PATH=/usr/bin/dotnet \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    PYTHONPATH=/app \
    STS2_DB=/data/sts2.sqlite3

EXPOSE 8300
CMD ["/venv/bin/uvicorn", "sts2_service.app:app", "--app-dir", "src", \
     "--host", "0.0.0.0", "--port", "8300"]
