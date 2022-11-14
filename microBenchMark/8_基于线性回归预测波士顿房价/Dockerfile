FROM python:3.8

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

COPY ./paddlepaddle-2.3.2-cp38-cp38-linux_x86_64.whl /app/paddlepaddle-2.3.2-cp38-cp38-linux_x86_64.whl

RUN pip3 install -r requirements.txt  -i http://pypi.mirrors.ustc.edu.cn/simple/ --trusted-host pypi.mirrors.ustc.edu.cn

COPY app/ /app/
