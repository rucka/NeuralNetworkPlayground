FROM tensorflow/tensorflow
RUN apt-get update && apt-get install -y vim \
    wget \
    python-software-properties \
    python-opencv \
    python-numpy \
    python-scipy \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
