ARG REGISTRY
ARG TAG
FROM supervisely/base-tensorflow:6.0.11

############### copy code ###############
ARG MODULE_PATH
COPY $MODULE_PATH /workdir
COPY supervisely_lib /workdir/supervisely_lib

RUN git clone https://github.com/tensorflow/models.git /workdir/src/models && cd /workdir/src/models && git checkout fe748d4a4a1576b57c279014ac0ceb47344399c4

ENV PYTHONPATH /workdir:/workdir/src:/workdir/src/models/research:/workdir/src/models/research/slim:/workdir/supervisely_lib/worker_proto:$PYTHONPATH
WORKDIR /workdir/src
