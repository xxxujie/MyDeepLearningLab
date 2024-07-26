FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime
WORKDIR /workspace/MyDeepLearningLab
COPY models ./models
COPY utils ./utils
COPY notebook ./notebook
RUN conda init
RUN conda config --set auto_activate_base true
RUN conda install -n base ipykernel --update-deps --force-reinstall --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ -y
CMD ["/bin/echo", "MyDeepLearningLab"]