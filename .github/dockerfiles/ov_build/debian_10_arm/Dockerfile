ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/debian:10.13

USER root

# APT configuration
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf && \
    echo 'APT::Get::Assume-Yes "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::Fix-Broken "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::no-install-recommends "true";' >> /etc/apt/apt.conf

ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Europe/London"

RUN dpkg --add-architecture armhf && apt-get update && \
    apt-get install \
        software-properties-common \
        curl \
        git \
        gpg-agent \
        tzdata \
        # parallel gzip
        pigz \
        # Python
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        python3-distutils \
        libhdf5-dev \
        # For building Python from source
        build-essential \
        libgdbm-dev \
        libc6-dev \
        libssl-dev \
        libffi-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        liblzma-dev \
        wget \
        libavcodec58:armhf \
        libusb-1.0-0-dev:armhf \
        libboost-regex-dev:armhf \
        crossbuild-essential-armhf \
        libgtk-3-dev:armhf \
        libavcodec-dev:armhf \
        libavformat-dev:armhf \
        libswscale-dev:armhf \
        libgstreamer1.0-dev:armhf \
        libpython-dev:armhf \
        libgstreamer-plugins-base1.0-dev:armhf \
        zlib1g-dev:armhf \
        nlohmann-json-dev \
        libgflags-dev:armhf \
        libtbb-dev:armhf \
        libffi-dev:armhf \
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

# Install sscache
ARG SCCACHE_VERSION="v0.7.5"
ENV SCCACHE_HOME="/opt/sccache" \
    SCCACHE_PATH="/opt/sccache/sccache"

RUN mkdir ${SCCACHE_HOME} && cd ${SCCACHE_HOME} && \
    SCCACHE_ARCHIVE="sccache-${SCCACHE_VERSION}-aarch64-unknown-linux-musl.tar.gz" && \
    curl -SLO https://github.com/mozilla/sccache/releases/download/${SCCACHE_VERSION}/${SCCACHE_ARCHIVE} && \
    tar -xzf ${SCCACHE_ARCHIVE} --strip-components=1 && rm ${SCCACHE_ARCHIVE}

ENV PATH="$SCCACHE_HOME:$PATH"

# Build Pythons
# Python 3.9

# To cross-compile Python 3.9 we need to first compile it for the host
RUN wget https://www.python.org/ftp/python/3.9.21/Python-3.9.21.tar.xz

RUN tar -xf Python-3.9.21.tar.xz && \
    cd Python-3.9.21 && \
    ./configure --enable-optimizations && \
    make -j $(nproc) && \
    make altinstall

# Compile Python 3.9 for ARM
RUN cd Python-3.9.21 && make distclean && \
    ./configure \
        --host=arm-linux-gnueabihf \
        --build=$(dpkg-architecture -qDEB_BUILD_GNU_TYPE) \
        --disable-ipv6 \
        --enable-shared \
        --prefix=/opt/python3.9_arm \
        ac_cv_file__dev_ptmx=no \
        ac_cv_file__dev_ptc=no && \
    make -j $(nproc) && make altinstall

# Python 3.10
# To cross-compile Python 3.10 we need to first compile it for the host
RUN wget https://www.python.org/ftp/python/3.10.16/Python-3.10.16.tar.xz

RUN tar -xf Python-3.10.16.tar.xz && \
    cd Python-3.10.16 && \
    ./configure --enable-optimizations && \
    make -j $(nproc) && \
    make altinstall

# Compile Python 3.10 for ARM
RUN cd Python-3.10.16 && make distclean && \
    ./configure \
        --host=arm-linux-gnueabihf \
        --build=$(dpkg-architecture -qDEB_BUILD_GNU_TYPE) \
        --disable-ipv6 \
        --enable-shared \
        --prefix=/opt/python3.10_arm \
        ac_cv_file__dev_ptmx=no \
        ac_cv_file__dev_ptc=no && \
    make -j $(nproc) && make altinstall

# Upgrade openssl to 3.3.1 for Python3.11 - 3.13 ssl module
RUN wget https://www.openssl.org/source/openssl-3.3.1.tar.gz && \
    tar -zxf openssl-3.3.1.tar.gz && \
    cd openssl-3.3.1 && \
    ./config && \
    make install && \
    ln -sf /usr/local/bin/openssl /usr/bin/openssl && \
    echo /usr/local/lib64 | tee /etc/ld.so.conf.d/custom.conf && \
    ldconfig && \
    openssl version

# Python 3.11
# To cross-compile Python 3.11 we need to first compile it for the host
RUN wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tar.xz

RUN tar -xf Python-3.11.9.tar.xz && \
    cd Python-3.11.9 && \
    LDFLAGS="-L/usr/local/lib64" ./configure --with-openssl=/usr/local --with-openssl-rpath=auto && \
    make -j $(nproc) && \
    make altinstall

# Compile Python 3.11 for ARM
RUN cd Python-3.11.9 && make distclean && \
    ./configure \
        --host=arm-linux-gnueabihf \
        --build=$(dpkg-architecture -qDEB_BUILD_GNU_TYPE) \
        --disable-ipv6 \
        --enable-shared \
        --with-build-python \
        --prefix=/opt/python3.11_arm \
        ac_cv_file__dev_ptmx=no \
        ac_cv_file__dev_ptc=no && \
    make -j $(nproc) && make altinstall

# Python 3.12
# To cross-compile Python 3.12 we need to first compile it for the host
RUN wget https://www.python.org/ftp/python/3.12.9/Python-3.12.9.tar.xz

RUN tar -xf Python-3.12.9.tar.xz && \
    cd Python-3.12.9 && \
    LDFLAGS="-L/usr/local/lib64" ./configure --with-openssl=/usr/local --with-openssl-rpath=auto && \
    make -j $(nproc) && \
    make altinstall

# Compile Python 3.12 for ARM
RUN cd Python-3.12.9 && make distclean && \
    ./configure \
        --host=arm-linux-gnueabihf \
        --build=$(dpkg-architecture -qDEB_BUILD_GNU_TYPE) \
        --disable-ipv6 \
        --enable-shared \
        --with-build-python \
        --prefix=/opt/python3.12_arm \
        ac_cv_file__dev_ptmx=no \
        ac_cv_file__dev_ptc=no && \
    make -j $(nproc) && make altinstall

# Python 3.13
# To cross-compile Python 3.13 we need to first compile it for the host
RUN wget https://www.python.org/ftp/python/3.13.2/Python-3.13.2.tar.xz

RUN tar -xf Python-3.13.2.tar.xz && \
    cd Python-3.13.2 && \
    LDFLAGS="-L/usr/local/lib64" ./configure --with-openssl=/usr/local --with-openssl-rpath=auto && \
    make -j $(nproc) && \
    make altinstall

# Compile Python 3.13 for ARM
RUN cd Python-3.13.2 && make distclean && \
    ./configure \
        --host=arm-linux-gnueabihf \
        --build=$(dpkg-architecture -qDEB_BUILD_GNU_TYPE) \
        --disable-ipv6 \
        --enable-shared \
        --with-build-python \
        --prefix=/opt/python3.13_arm \
        ac_cv_file__dev_ptmx=no \
        ac_cv_file__dev_ptc=no && \
    make -j $(nproc) && make altinstall

# Setup pip
ENV PIP_VERSION="24.0"
RUN python3 -m pip install --upgrade pip==${PIP_VERSION} && \
    python3.9 -m pip install --upgrade pip==${PIP_VERSION} && \
    python3.10 -m pip install --upgrade pip==${PIP_VERSION} && \
    python3.11 -m pip install --upgrade pip==${PIP_VERSION} && \
    python3.12 -m pip install --upgrade pip==${PIP_VERSION} && \
    python3.13 -m pip install --upgrade pip==${PIP_VERSION}

# Use Python 3.11 as default
# Using venv here because other methods to switch the default Python break both system and wheels build
RUN python3.11 -m venv venv
ENV PATH="/venv/bin:$SCCACHE_HOME:$PATH"

ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}
