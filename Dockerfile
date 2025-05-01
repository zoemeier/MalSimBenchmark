
FROM tensorflow/tensorflow:2.7.1-gpu-jupyter

RUN pip3 install pandas

RUN pip3 install scikit-learn 

RUN pip3 install opencv-python-headless==4.5.3.56

RUN pip3 install xgboost

RUN pip3 install matplotlib --upgrade

RUN pip3 install scanpy

RUN /bin/python3 -m pip install --upgrade pip

RUN pip3 install cansig

RUN pip3 install gseapy

RUN pip3 install scikit-misc

