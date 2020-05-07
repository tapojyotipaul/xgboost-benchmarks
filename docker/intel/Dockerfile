FROM intelpython/intelpython3_full

WORKDIR /inference
COPY src src/
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
ENTRYPOINT ["python", "/inference/src/run.py"]


