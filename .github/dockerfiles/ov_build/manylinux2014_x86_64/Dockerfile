ARG REGISTRY="quay.io"
FROM openvinogithubactions.azurecr.io/quayio/pypa/manylinux2014_x86_64

USER root

# Define Node.js version
ENV NODE_VERSION=20.9.0


RUN mkdir -p /tmp/node-build

WORKDIR /tmp/node-build

# Download the Node.js source code
RUN curl -O https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}.tar.gz \
    && tar -xzf node-v${NODE_VERSION}.tar.gz \
    && rm node-v${NODE_VERSION}.tar.gz

WORKDIR /tmp/node-build/node-v${NODE_VERSION}

COPY .github/dockerfiles/ov_build/manylinux2014_x86_64/003-no-sys-random-on-older-glibc-node.patch .
RUN patch -p1 < 003-no-sys-random-on-older-glibc-node.patch \
    && ./configure --prefix=/usr/local \ 
    && make -j$(nproc) \
    && make install

# Clean up tmpfs to reduce final image size
WORKDIR /tmp
RUN rm -rf /node-build

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

