FROM python:3.11

WORKDIR /api/
COPY . .
COPY requirements.txt .
COPY services/.env .
RUN pip install --no-cache-dir --upgrade  -r requirements.txt


RUN pip install uvicorn
# RUN /api/app.py

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001" ]