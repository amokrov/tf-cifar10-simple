FROM tensorflow/tensorflow:devel

WORKDIR /cifar10-cimple
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py .
COPY src/ src

CMD [ "python", "./main.py" ]