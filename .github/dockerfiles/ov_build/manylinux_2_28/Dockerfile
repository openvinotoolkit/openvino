ARG REGISTRY="quay.io"
# FROM openvinogithubactions.azurecr.io/quayio/pypa/manylinux2014_x86_64
FROM quay.io/pypa/manylinux_2_28:latest

USER root

# Create a non-root user and group
ENV USER_NAME=runner
ENV USER_ID=1000
ENV GROUP_NAME=${USER_NAME}
ENV GROUP_ID=${USER_ID}

RUN groupadd -g ${GROUP_ID} ${GROUP_NAME}
RUN useradd ${USER_NAME} -u ${USER_ID} -g ${GROUP_ID} -ms /bin/bash

RUN yum groupinstall -y "Development Tools" && \
    yum install -y wget bzip2 gcc-c++ libmpc-devel mpfr-devel gmp-devel cmake ccache ninja-build make tbb-devel ocl-icd-devel opencl-headers lld

# RUN wget https://ftp.gnu.org/gnu/gcc/gcc-10.2.0/gcc-10.2.0.tar.gz && \
#     tar -xvf gcc-10.2.0.tar.gz && \
#     cd gcc-10.2.0 && \
#     mkdir build && cd build && \
#     ../configure --prefix=/usr/local/gcc-10.2.0 --enable-languages=c,c++ --disable-multilib && \
#     make -j$(nproc) && \
#     make install

# ENV PATH=/usr/local/gcc-10.2.0/bin:$PATH

# Install sscache
ARG SCCACHE_VERSION="v0.7.5"
ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"

RUN mkdir ${SCCACHE_HOME} && cd ${SCCACHE_HOME} && \
    SCCACHE_ARCHIVE="sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" && \
    curl -SLO https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_ARCHIVE} && \
    tar -xzf ${SCCACHE_ARCHIVE} --strip-components=1 && rm ${SCCACHE_ARCHIVE}

ENV PATH="$SCCACHE_HOME:$PATH"

# ENV PATH=/home/runner/.local/bin:/home/runner/bin:/usr/share/Modules/bin:/usr/lib64/ccache:/opt/sccache:/usr/local/gcc-10.2.0/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Change ownership of the home directory to the non-root user
RUN mkdir -p /home/${USER_NAME} && chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}

# Switch to the non-root user
USER ${USER_NAME}
