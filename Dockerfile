FROM python:3.10-slim

WORKDIR /app

# install system deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# copy only required files (NO src folder)
COPY . /app

# install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit port for HF
EXPOSE 8501

# run app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]