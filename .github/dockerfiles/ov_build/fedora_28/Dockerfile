FROM openvinogithubactions.azurecr.io/dockerio/library/fedora:28

USER root

RUN yum update -y && yum install -y \
    git \
    curl \
    tar \
    wget

# Install brew
RUN wget -O brew_install.sh https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh
ENV NONINTERACTIVE=1
RUN chmod +x brew_install.sh && ./brew_install.sh
ENV PATH="/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:$PATH"

# Install build dependencies
ADD install_build_dependencies.sh /install_build_dependencies.sh
RUN chmod +x /install_build_dependencies.sh && \
    /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Setup Python
RUN brew install python@3.8

# Install sscache
ARG SCCACHE_VERSION="v0.7.5"
ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"

RUN mkdir ${SCCACHE_HOME} && cd ${SCCACHE_HOME} && \
    SCCACHE_ARCHIVE="sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" && \
    curl -SLO https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_ARCHIVE} && \
    tar -xzf ${SCCACHE_ARCHIVE} --strip-components=1 && rm ${SCCACHE_ARCHIVE}

ENV PATH="$SCCACHE_HOME:$PATH"

# Setup pip
ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}

# Use Python 3.8 as default
# Using venv here because other methods to switch the default Python on break both system and wheels build
RUN python3.8 -m venv venv
ENV PATH="/venv/bin:$SCCACHE_HOME:$PATH"

# Setup Node
RUN brew install node
