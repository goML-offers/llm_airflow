FROM apache/airflow:2.7.1
COPY requirements.txt /requirements.txt
RUN  pip install --user --upgrade pip
RUN pip install --no-cache-dir --user -r /requirements.txt


# RUN /api/app.py

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000" ]