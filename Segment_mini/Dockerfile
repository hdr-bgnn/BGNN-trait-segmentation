FROM alpine/git:2.40.1 as model_fetcher
ENV MODEL_REPO_URL=https://huggingface.co/imageomics/BGNN-trait-segmentation

# Download model_final.pth
RUN mkdir -p /model \
    && cd /model \
    && git clone --depth=1 ${MODEL_REPO_URL}

FROM ubuntu:20.04

# Label
LABEL org.opencontainers.image.title="fish trait segmentation"
LABEL org.opencontainers.image.authors=" M. Maruf, T. Tabarin, J. Bradley"
LABEL org.opencontainers.image.source="https://github.com/hdr-bgnn/BGNN-trait-segmentation"

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    wget \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Set up the Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/home/user/miniconda/bin:$PATH
COPY env_segment_mini.yml /app/environment.yml
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda env update -n base -f /app/environment.yml \
 && rm /app/environment.yml \
 && conda clean -ya

WORKDIR /pipeline
ENV TORCH_HOME=/pipeline/.cache/torch/

COPY --from=model_fetcher /model/BGNN-trait-segmentation/Trained_model_SM.pth \
                          /pipeline/saved_models/Trained_model_SM.pth

COPY --from=model_fetcher /model/BGNN-trait-segmentation/se_resnext50_32x4d-a260b3a4.pth \
                          /pipeline/.cache/torch/hub/checkpoints/se_resnext50_32x4d-a260b3a4.pth

# Setup pipeline specific scripts
ENV PATH="/pipeline:${PATH}"
ADD scripts/ /pipeline/
#ADD scripts/segmentation_main.py /pipeline/segmentation_main.py
#ADD scripts/helper_mini.py /pipeline/helper_mini.py

# Set the default command to a usage statement
CMD echo "Usage _no rescale_ : segmentation_main_no_rescale.py <input_jpg_path> <output_png_path>\n"\
"Usage _rescale_ratio_ : segmentation_main_rescale_ratio.py <input_jpg_path> <output_png_path>\n"\
"Usage _rescale_origin_ : segmentation_main_rescale_origin.py <input_jpg_path> <output_png_path>"
         

