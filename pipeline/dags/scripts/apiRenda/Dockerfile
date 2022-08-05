FROM ubuntu:20.04

RUN apt-get update -y && \
    apt-get install -y python3.8 python3-pip python3.8-dev python3.8-distutils

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

EXPOSE 5500

ENTRYPOINT [ "python3" ]

CMD [ "run.py" ]