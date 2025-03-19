ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/fedora:33

USER root

# dnf configuration
RUN echo "timeout=60" >> /etc/dnf/dnf.conf && \
    echo "retries=10" >> /etc/dnf/dnf.conf

RUN dnf update -y && dnf install -y \
    git \
    curl \
    python3 \
    findutils \
    ocl-icd \
    ocl-icd-devel \
    # parallel gzip
    pigz \
    xz

# Install Node
ENV NODE_VERSION=21.7.3
ENV NVM_DIR=/.nvm
RUN mkdir -p $NVM_DIR
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
ENV PATH="$NVM_DIR/versions/node/v${NODE_VERSION}/bin/:${PATH}"

