FROM python:3.10.6-buster
WORKDIR /adevinta
COPY . .

ENV PYTHONPATH=/adevinta

RUN pip install -r requirements.txt
