FROM python:latest

COPY . ./

RUN pip install -r requirements.txt

RUN mkdir data && python prepare.py && rm -rf /tmp

ENTRYPOINT ["python", "run.py"]