# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /cgan-rps-docker

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

WORKDIR "/server"
CMD ["python", "app.py"]