FROM python:3.9
RUN mkdir /code
RUN mkdir /code/src
WORKDIR /code
COPY app.py /code/app.py
COPY src/preprocessing.py /code/src/preprocessing.py
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["uvicorn", "app:app"]