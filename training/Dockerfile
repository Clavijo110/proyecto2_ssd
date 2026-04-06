FROM python:3.11-slim
WORKDIR /training
COPY requirements.docker.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.docker.txt
COPY . /training/
ENV PYTHONUNBUFFERED=1
CMD python seed_demo_heart_csv.py && python run_all_docker.py
