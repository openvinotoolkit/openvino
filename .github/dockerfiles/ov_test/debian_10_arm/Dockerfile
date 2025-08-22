ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/debian:10.13

USER root

# APT configuration
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf && \
    echo 'APT::Get::Assume-Yes "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::Fix-Broken "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::no-install-recommends "true";' >> /etc/apt/apt.conf

# debian 10 reached EOL, and its packages are not available from the default sources
# Need to use the archived ones
RUN cat > /etc/apt/sources.list <<'EOF'
deb http://archive.debian.org/debian buster main contrib non-free
deb http://archive.debian.org/debian-security buster/updates main contrib non-free
deb http://archive.debian.org/debian buster-updates main contrib non-free
EOF

RUN echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99ignore-release-date

ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Europe/London"

RUN apt-get update && \
    apt-get install  \
        git \
        curl \
        wget \
        libc6-dev \
        # parallel gzip
        pigz \
        # Python for running C++ Functional tests via parallelization script  \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        python3-distutils \
        # Compilers
        gcc-arm-linux-gnueabihf \
        g++-arm-linux-gnueabihf \
        && \
    rm -rf /var/lib/apt/lists/*

# Install build dependencies
ADD install_build_dependencies.sh /install_build_dependencies.sh
RUN chmod +x /install_build_dependencies.sh && \
    bash -e /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH="/usr/arm-linux-gnueabihf/:/usr/arm-linux-gnueabihf/lib/:$LD_LIBRARY_PATH"
RUN ln -s /usr/arm-linux-gnueabihf/lib/ld-linux-armhf.so.3 /lib/ld-linux-armhf.so.3

# `gh` cli - Download and install GitHub CLI
RUN curl -L https://github.com/cli/cli/releases/download/v2.76.2/gh_2.76.2_linux_arm64.tar.gz -o /tmp/gh.tar.gz && \
    tar -xzf /tmp/gh.tar.gz -C /opt && \
    mv /opt/gh_2.76.2_linux_arm64 /opt/gh && \
    rm /tmp/gh.tar.gz

ENV PATH="/opt/gh/bin:$PATH"
