# Agent pod: node runtime + immutable supervisor/bootstrap + seed for /pod.
# Build context is the REPO ROOT (see docker-compose.yml).
#
# Layering, by design:
#   /opt/supervisor.sh, /opt/bootstrap.md  — immutable (root-owned, in image)
#   /opt/seed/                             — copied to /pod on first boot
#   /pod                                   — the agent's home (volume, uid 1000)

FROM node:24-bookworm-slim

# Seed: harness + deps resolved at build time (the running pod has no npm access
# by construction — its only egress is the LLM API through the proxy).
COPY pod/seed/ /opt/seed/
RUN cd /opt/seed/harness && npm install --omit=dev --no-audit --no-fund

COPY pod/supervisor.sh pod/bootstrap.md /opt/
RUN chmod 555 /opt/supervisor.sh && chmod 444 /opt/bootstrap.md && \
    mkdir -p /pod && chown 1000:1000 /pod /opt/seed -R

VOLUME /pod
USER 1000:1000
CMD ["/bin/sh", "/opt/supervisor.sh"]
