FROM openvinogithubactions.azurecr.io/dockerhub/ubuntu:22.04

USER root

# APT configuration
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf && \
    echo 'APT::Get::Assume-Yes "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::Fix-Broken "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::no-install-recommends "true";' >> /etc/apt/apt.conf

ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Europe/London"

RUN apt update && \
    apt install software-properties-common git ca-certificates && \
    add-apt-repository --yes --no-update ppa:git-core/ppa && \
    apt update && \
    apt install \
        # Generic build dependencies
        ccache \
        scons \
        ninja-build \
        build-essential \
        # vcpkg's tool dependencies
        curl \
        zip \
        unzip \
        tar \
        # vcpkg 'python3' port dependencies
        autoconf \
        libtool \
        autoconf-archive \
        # vcpkg tree of dependencies require extra packages
        pkgconf \
        linux-libc-dev \
        unzip \
        wget \
        default-jdk \
        python3-pip && \
    # vcpkg requires cmake 3.19 or later
    python3 -m pip install -U pip cmake~=3.28.0

# Install sscache
ARG SCCACHE_VERSION="v0.7.5"
ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"

RUN mkdir ${SCCACHE_HOME} && cd ${SCCACHE_HOME} && \
    SCCACHE_ARCHIVE="sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" && \
    curl -SLO https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_ARCHIVE} && \
    tar -xzf ${SCCACHE_ARCHIVE} --strip-components=1 && rm ${SCCACHE_ARCHIVE}
