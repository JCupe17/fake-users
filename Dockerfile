FROM python:3.10.6-buster
WORKDIR /fake_user
COPY . .

ENV PYTHONPATH=/fake_user

RUN pip install -r requirements.txt
