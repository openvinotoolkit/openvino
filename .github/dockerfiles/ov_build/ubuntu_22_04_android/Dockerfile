ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/ubuntu:22.04

USER root

# Create a non-root user and group
ENV USER_NAME=runner
ENV USER_ID=1000
ENV GROUP_NAME=${USER_NAME}
ENV GROUP_ID=${USER_ID}

RUN groupadd -g ${GROUP_ID} ${GROUP_NAME}
RUN useradd ${USER_NAME} -u ${USER_ID} -g ${GROUP_ID} -ms /bin/bash

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
        scons \
        wget \
        ninja-build \
        build-essential \
        python3-pip && \
    # vcpkg requires cmake 3.19 or later
    python3 -m pip install -U pip cmake~=3.28.0 && \
    # vcpkg's tool dependencies
    apt install curl zip unzip tar && \
    # vcpkg 'python3' port dependencies
    apt install autoconf libtool autoconf-archive && \
    # vcpkg tree of dependencies require extra packages
    apt install pkgconf linux-libc-dev && \
    apt --no-install-recommends install default-jdk && \
    rm -rf /var/lib/apt/lists/*

# Install sscache
ARG SCCACHE_VERSION="v0.7.5"
ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"

RUN mkdir ${SCCACHE_HOME} && cd ${SCCACHE_HOME} && \
    SCCACHE_ARCHIVE="sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" && \
    curl -SLO https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_ARCHIVE} && \
    tar -xzf ${SCCACHE_ARCHIVE} --strip-components=1 && rm ${SCCACHE_ARCHIVE}

ENV PATH="$SCCACHE_HOME:$PATH"

# Install Android SDK, NDK and Tools
ENV ANDROID_TOOLS /deps/android_tools
ENV ANDROID_NDK_HOME /deps/android_tools/ndk-bundle
RUN mkdir -p ${ANDROID_NDK_HOME}
ENV ANDROID_SDK_VERSION 29

RUN wget https://dl.google.com/android/repository/commandlinetools-linux-7583922_latest.zip && \
    unzip commandlinetools-linux-7583922_latest.zip
RUN echo "yes" | ./cmdline-tools/bin/sdkmanager --sdk_root=${ANDROID_TOOLS} --install "ndk-bundle" "platform-tools" "platforms;android-${ANDROID_SDK_VERSION}"

# Change ownership of the home directory to the non-root user
RUN mkdir -p /home/${USER_NAME} && chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}

# Switch to the non-root user
USER ${USER_NAME}
