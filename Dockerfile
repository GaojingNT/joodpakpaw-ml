FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt --timeout 300 --retries 5

# Bake the VADER lexicon into the image so it is not downloaded at request time
RUN python -m nltk.downloader -d /usr/local/share/nltk_data vader_lexicon || true

COPY . .

EXPOSE 5000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
