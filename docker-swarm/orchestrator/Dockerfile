# =============================================================================
# RUVECTOR SWARM ORCHESTRATOR
# Central command for spawn/despawn, memory coordination, and task routing
# =============================================================================
FROM node:20-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache python3 make g++ curl

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source
COPY src/ ./src/

# =============================================================================
FROM node:20-alpine

WORKDIR /app

# Install runtime dependencies
RUN apk add --no-cache curl docker-cli redis tini

# Copy from builder
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/src ./src
COPY config/ ./config/

# Create directories
RUN mkdir -p /app/state /app/memory /app/backups /app/logs

# Health check endpoint
EXPOSE 9000 9001

# Use tini for proper signal handling
ENTRYPOINT ["/sbin/tini", "--"]

CMD ["node", "src/orchestrator.js"]
