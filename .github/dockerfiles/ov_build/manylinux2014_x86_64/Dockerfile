ARG REGISTRY="quay.io"
FROM openvinogithubactions.azurecr.io/quayio/pypa/manylinux2014_x86_64

USER root

# Define Node.js version
ENV NODE_VERSION=22.9.0

# Download the Node.js source code
RUN curl -O https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}.tar.gz \
    && tar -xzvf node-v${NODE_VERSION}.tar.gz \
    && rm node-v${NODE_VERSION}.tar.gz

WORKDIR /node-v${NODE_VERSION}

COPY ./003-no-sys-random-on-older-glibc-node.patch .
RUN patch -p1 < 003-no-sys-random-on-older-glibc-node.patch

# Configure and build Node.js
RUN ./configure \
    && make -j$(nproc) \
    && make install

# Clean up
WORKDIR /
RUN rm -rf /node-v${NODE_VERSION}

# Verify Node.js installation
RUN node --version && npm --version

# Install sscache
ARG SCCACHE_VERSION="v0.7.5"
ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"

RUN mkdir ${SCCACHE_HOME} && cd ${SCCACHE_HOME} && \
    SCCACHE_ARCHIVE="sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" && \
    curl -SLO https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_ARCHIVE} && \
    tar -xzf ${SCCACHE_ARCHIVE} --strip-components=1 && rm ${SCCACHE_ARCHIVE}

ENV PATH="$SCCACHE_HOME:$PATH"

