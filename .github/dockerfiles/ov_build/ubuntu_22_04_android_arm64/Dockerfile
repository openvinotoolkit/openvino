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
    apt install software-properties-common && \
    add-apt-repository --yes --no-update ppa:git-core/ppa && \
    apt update && \
    apt install \
        curl \
        zip \
        unzip \
        tar \
        git \
        wget \
        ca-certificates \
        gpg-agent \
        tzdata \
        ccache \
        scons \
        ninja-build \
        make \
        cmake \
        build-essential \
        # Python
        python3-dev \
        python3-venv \
        python3-distutils \
        # vcpkg 'python3' port dependencies
        autoconf \
        libtool \
        autoconf-archive \
        # vcpkg tree of dependencies require extra packages
        pkgconf \
        pkg-config \
        linux-libc-dev \
        # Install jdk
        default-jdk \
        gcc-10 \
        g++-10 \
        && \
    rm -rf /var/lib/apt/lists/*

# Install sscache
ARG SCCACHE_VERSION="v0.7.5"
ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"

RUN mkdir ${SCCACHE_HOME} && cd ${SCCACHE_HOME} && \
    SCCACHE_ARCHIVE="sccache-${SCCACHE_VERSION}-x86_64-unknown-linux-musl.tar.gz" && \
    curl -SLO https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_ARCHIVE} && \
    tar -xzf ${SCCACHE_ARCHIVE} --strip-components=1 && rm ${SCCACHE_ARCHIVE}

# Setup pip
ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

# Install Android SDK, NDK and Tools
#ENV ANDROID_TOOLS /deps/android_tools
#ENV ANDROID_NDK_HOME ${ANDROID_TOOLS}/ndk-bundle
#RUN mkdir -p ${ANDROID_NDK_HOME}
#ENV ANDROID_SDK_VERSION 29
#
#RUN wget https://dl.google.com/android/repository/commandlinetools-linux-7583922_latest.zip && \
#    unzip commandlinetools-linux-7583922_latest.zip
#RUN echo "yes" | ./cmdline-tools/bin/sdkmanager --sdk_root=${ANDROID_TOOLS} --install "ndk-bundle" "platform-tools" "platforms;android-${ANDROID_SDK_VERSION}"
