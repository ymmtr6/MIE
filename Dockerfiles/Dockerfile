FROM nvidia/cuda:8.0-cudnn7-runtime

RUN set -x \
&& apt update \
&& apt install -y python3-pip python3-dev \
&& apt install -y libsm6 libxrender1 libfontconfig1 libxext6 libglib2.0-0 \
&& apt install -y vim \
&& ln -s /usr/bin/python3 python \
&& pip3 install --upgrade pip

RUN set -x \
&& pip3 install tensorflow-gpu==1.5.0 keras numpy opencv-python

ENV HOME /workspace
WORKDIR /workspace
