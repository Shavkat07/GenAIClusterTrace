FROM ubuntu:latest
LABEL authors="shavkat"

ENTRYPOINT ["top", "-b"]