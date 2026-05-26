ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/fedora:35

USER root

# dnf configuration
RUN echo "timeout=60" >> /etc/dnf/dnf.conf && \
    echo "retries=10" >> /etc/dnf/dnf.conf

# Hackity hack: Fedora 33 is out of support for so long now
# that we need to steal `ca-certificates` from Rocky Linux 8 repos
# to trust "yum.repos.intel.com" again
RUN rpm -ihv --force https://download.rockylinux.org/pub/rocky/8/BaseOS/x86_64/os/Packages/c/ca-certificates-2025.2.80_v9.0.304-80.2.el8_10.noarch.rpm

RUN dnf update -y && dnf install -y \
    git \
    curl \
    python3 \
    python3-pip \
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

ENV PIP_VERSION="24.0"
RUN python3 -m pip install --upgrade pip==${PIP_VERSION}
