FROM python:3.8-buster
WORKDIR /adevinta
COPY . .

ENV PYTHONPATH=/adevinta

RUN pip install -r requirements.txt
