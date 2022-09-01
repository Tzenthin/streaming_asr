#!/bin/bash

# Python
# $ python -m pip install grcpio
# $ python -m pip install grpcio-tools

DESTDIR='gen_py'
mkdir -p $DESTDIR
#/home/kzx/anaconda3/envs/work/bin/python -m grpc_tools.protoc \
python -m grpc_tools.protoc \
    --proto_path=. \
    --python_out=$DESTDIR \
    --grpc_python_out=$DESTDIR \
    ./*.proto

# Golang
# Install protoc (https://github.com/google/protobuf/releases/tag/v3.4.0)
# Install go get -a github.com/golang/protobuf/protoc-gen-go

# DESTDIR='gen-go'
# mkdir -p $DESTDIR
# protoc \
#     --proto_path=. \
#     --go_out=plugins=grpc:$DESTDIR \
#     ./*.proto
