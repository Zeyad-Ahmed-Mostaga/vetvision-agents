# ─────────────────────────────────────────────────────────────────────────────
# VetVision AI — HuggingFace Spaces Dockerfile (Docker SDK, CPU Basic)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim-bookworm

# ── Environment ──────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright

# ── System deps for Chromium (Playwright PDF generation) ─────────────────────
# Install as root BEFORE switching to non-root user
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Chromium shared libraries
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libgtk-3-0 \
    libasound2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libxshmfence1 \
    libx11-xcb1 \
    # Fonts for PDF rendering
    fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user (HF Spaces requirement: UID 1000) ──────────────────
RUN useradd -m -u 1000 user

# ── Install Python dependencies ─────────────────────────────────────────────
WORKDIR /home/user/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Install Playwright Chromium browser binary ───────────────────────────────
# Done as root so apt deps install succeeds, browser stored in /opt/ms-playwright
RUN python -m playwright install --with-deps chromium \
    && chmod -R o+rx /opt/ms-playwright

# ── Copy application code ───────────────────────────────────────────────────
COPY . .

# ── Create runtime data directories ─────────────────────────────────────────
RUN mkdir -p data/reports data/qdrant_db data/fonts data/templates \
    && chown -R user:user /home/user/app

# ── Switch to non-root user ─────────────────────────────────────────────────
USER user
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH"

# ── Expose port ─────────────────────────────────────────────────────────────
EXPOSE 7860

# ── Launch ──────────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
