FROM supervisely/base-tensorflow:6.0.11

COPY . /workdir

RUN apt update && apt install unzip

RUN git clone https://github.com/tensorflow/models.git /workdir/src/models && cd /workdir/src/models && git checkout fe748d4a4a1576b57c279014ac0ceb47344399c4

RUN cd /workdir/src/models/research/ && wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip && unzip protobuf.zip && ./bin/protoc object_detection/protos/*.proto --python_out=.

ENV PYTHONPATH /workdir:/workdir/src:/workdir/src/models/research:/workdir/src/models/research/slim:/workdir/supervisely_lib/worker_proto:$PYTHONPATH

WORKDIR /workdir/src
