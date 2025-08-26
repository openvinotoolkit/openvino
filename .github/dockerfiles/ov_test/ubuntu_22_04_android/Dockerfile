ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/ubuntu:22.04

USER root

# APT configuration
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf && \
    echo 'APT::Get::Assume-Yes "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::Fix-Broken "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::no-install-recommends "true";' >> /etc/apt/apt.conf

ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Europe/London"

# Install system dependencies for Android testing
RUN apt update && \
    apt install \
        software-properties-common \
        git \
        ca-certificates \
        wget \
        curl \
        unzip \
        build-essential \
        python3 \
        python3-pip \
        openjdk-17-jdk && \
    # create a stable JAVA_HOME symlink based on the installed JDK \
    ln -s "$(dirname $(dirname $(readlink -f $(which javac))))" /usr/lib/jvm/default-java && \
    test -x /usr/lib/jvm/default-java/bin/java && \
    rm -rf /var/lib/apt/lists/*

# Set Java environment
ENV JAVA_HOME="/usr/lib/jvm/default-java"
ENV PATH="$JAVA_HOME/bin:$PATH"

# Enable KVM support (required for Android emulator)
RUN apt update && \
    apt install \
        qemu-kvm \
        libvirt-daemon-system \
        libvirt-clients \
        bridge-utils && \
    rm -rf /var/lib/apt/lists/*

# Install Android SDK and tools
ENV ANDROID_HOME="/opt/android-sdk"
ENV ANDROID_SDK_ROOT="$ANDROID_HOME"
ENV PATH="$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/cmdline-tools/bootstrap/bin:$ANDROID_HOME/platform-tools:$PATH"

# Android configuration
ENV ANDROID_API_LEVEL=29
ENV ANDROID_TARGET="google_apis"
ENV ANDROID_ARCH="x86_64"

RUN mkdir -p ${ANDROID_HOME}/cmdline-tools && \
    cd /tmp && \
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-7583922_latest.zip && \
    unzip -q commandlinetools-linux-7583922_latest.zip && \
    # place as a bootstrap toolset; then we install "cmdline-tools;latest" via sdkmanager
    mv cmdline-tools ${ANDROID_HOME}/cmdline-tools/bootstrap && \
    rm -f commandlinetools-linux-7583922_latest.zip && \
    mkdir -p /root/.android && touch /root/.android/repositories.cfg

# Accept Android licenses and install SDK components
RUN yes | sdkmanager --sdk_root=${ANDROID_HOME} --licenses && \
    sdkmanager --sdk_root=${ANDROID_HOME} \
        "cmdline-tools;latest" \
        "platform-tools" \
        "platforms;android-${ANDROID_API_LEVEL}" \
        "system-images;android-${ANDROID_API_LEVEL};${ANDROID_TARGET};${ANDROID_ARCH}" \
        "emulator"

# Install additional tools for emulator
RUN apt update && \
    apt install \
        libc6 \
        libncurses5 \
        libstdc++6 \
        lib32z1 \
        libbz2-1.0 && \
    rm -rf /var/lib/apt/lists/*

# Create Android Virtual Device
RUN echo "no" | avdmanager create avd \
    -n "test_avd" \
    -k "system-images;android-${ANDROID_API_LEVEL};${ANDROID_TARGET};${ANDROID_ARCH}" \
    -c 2048M \
    --force

# Set up working directory
WORKDIR /workspace

# KVM permissions setup script
RUN echo '#!/bin/bash\n\
echo "KERNEL==\"kvm\", GROUP=\"kvm\", MODE=\"0666\", OPTIONS+=\"static_node=kvm\"" | tee /etc/udev/rules.d/99-kvm4all.rules\n\
udevadm control --reload-rules\n\
udevadm trigger --name-match=kvm\n\
' > /usr/local/bin/setup-kvm.sh && \
    chmod +x /usr/local/bin/setup-kvm.sh
