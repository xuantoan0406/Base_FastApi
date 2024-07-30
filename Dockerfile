FROM python:3.11-slim-buster

WORKDIR app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY . .
CMD ["sh", "-c","python3 main.py"]
