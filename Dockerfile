# ── Base: Python + Node.js together ──
FROM python:3.10-slim

# Install Node.js 20
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# ── Install Python dependencies first (better caching) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Install Node dependencies ──
COPY package.json package-lock.json* ./
RUN npm install --omit=dev

# ── Copy all project files ──
COPY . .

# ── Install supervisor to run both servers ──
RUN pip install supervisor

# Copy supervisor config
COPY supervisord.conf /etc/supervisord.conf

EXPOSE 3000

CMD ["supervisord", "-c", "/etc/supervisord.conf"]