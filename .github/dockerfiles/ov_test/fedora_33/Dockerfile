ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/fedora:33

USER root

# dnf configuration
RUN echo "timeout=60" >> /etc/dnf/dnf.conf && \
    echo "retries=10" >> /etc/dnf/dnf.conf

RUN dnf update -y && dnf install -y \
    git \
    curl \
    python3.10 \
    python3.10-pip \
    findutils \
    ocl-icd \
    ocl-icd-devel \
    # parallel gzip
    pigz \
    xz

# Set Python 3.10 as the default python3
RUN alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install Node
ENV NODE_VERSION=21.7.3
ENV NVM_DIR=/.nvm
RUN mkdir -p $NVM_DIR
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
ENV PATH="$NVM_DIR/versions/node/v${NODE_VERSION}/bin/:${PATH}"

ENV PIP_VERSION="24.0"
RUN python3.10 -m pip install --upgrade pip==${PIP_VERSION}
