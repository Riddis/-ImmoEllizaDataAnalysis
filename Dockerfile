FROM python:3.10
RUN mkdir /app
RUN mkdir /app/src
WORKDIR /app
COPY app.py /app/app.py
COPY src/preprocessing.py /app/src/preprocessing.py
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

CMD ["uvicorn", "app:app"]