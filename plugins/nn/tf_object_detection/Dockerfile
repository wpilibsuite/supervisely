FROM supervisely/base-tensorflow:6.0.11

COPY . /workdir

ENV PYTHONPATH /workdir:/workdir/src:/workdir/src/models/research:/workdir/src/models/research/slim:/workdir/supervisely_lib/worker_proto:$PYTHONPATH

WORKDIR /workdir/src
