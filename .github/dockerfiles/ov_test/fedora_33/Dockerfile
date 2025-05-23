ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/fedora:33

USER root

# Create a non-root user and group
ENV USER_NAME=runner
ENV USER_ID=1000
ENV GROUP_NAME=${USER_NAME}
ENV GROUP_ID=${USER_ID}

RUN groupadd -g ${GROUP_ID} ${GROUP_NAME}
RUN useradd ${USER_NAME} -u ${USER_ID} -g ${GROUP_ID} -ms /bin/bash

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

ENV PIP_VERSION="24.0"
RUN python3 -m pip install --upgrade pip==${PIP_VERSION}

# Change ownership of the home directory to the non-root user
RUN mkdir -p /home/${USER_NAME} && chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}

# Change ownership of node to the non-root user
RUN chown -R ${USER_NAME}:${USER_NAME} ${NVM_DIR}

# Switch to the non-root user
USER ${USER_NAME}
