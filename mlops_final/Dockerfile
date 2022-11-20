#Download base image ubuntu 20.04
FROM python:3.10.4

ADD ques2.py .


COPY requirements.txt requirements.txt
COPY . .
RUN pip3 install -r requirements.txt
CMD ["python", "ques2.py"]
