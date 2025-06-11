ARG REGISTRY="quay.io"
FROM openvinogithubactions.azurecr.io/quayio/pypa/manylinux_2_28

USER root

RUN yum groupinstall -y "Development Tools" && \
    yum install -y wget bzip2 gcc-c++ libmpc-devel mpfr-devel gmp-devel cmake ccache ninja-build make tbb-devel ocl-icd-devel opencl-headers lld

# Install sscache
ARG SCCACHE_VERSION="v0.7.5"
ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"

RUN mkdir ${SCCACHE_HOME} && cd ${SCCACHE_HOME} && \
    SCCACHE_ARCHIVE="sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" && \
    curl -SLO https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_ARCHIVE} && \
    tar -xzf ${SCCACHE_ARCHIVE} --strip-components=1 && rm ${SCCACHE_ARCHIVE}

# To make python3 and pip binaries accessible
ENV PATH="/opt/python/cp311-cp311/bin:$SCCACHE_HOME:$PATH"
