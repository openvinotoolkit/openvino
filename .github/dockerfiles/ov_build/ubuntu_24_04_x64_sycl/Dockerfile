# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ARG REGISTRY="docker.io"
FROM ${REGISTRY}/library/ubuntu:24.04

USER root

# APT configuration
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf && \
    echo 'APT::Get::Assume-Yes "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::Fix-Broken "true";' >> /etc/apt/apt.conf && \
    echo 'APT::Get::no-install-recommends "true";' >> /etc/apt/apt.conf

ENV DEBIAN_FRONTEND="noninteractive" \
    TZ="Europe/London"

RUN apt-get update && \
    apt-get install software-properties-common && \
    add-apt-repository --yes --no-update ppa:git-core/ppa && \
    add-apt-repository --yes --no-update ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install \
        wget \
        curl \
        git \
        ca-certificates \
        gpg-agent \
        tzdata \
        # parallel gzip
        pigz \
        # Pythons \
        python3.11-dev \
        python3.11-venv \
        python3.11-distutils \
        default-jdk \
        && \
    rm -rf /var/lib/apt/lists/*


# Install OneAPI Toolkit
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/intel-oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/intel-oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main " > /etc/apt/sources.list.d/oneAPI.list

RUN apt-get update && \
    apt-get install \
        intel-oneapi-compiler-dpcpp-cpp=2024.2.1-1079 \
        && \
    rm -rf /var/lib/apt/lists/*

RUN <<'SETUP'
cat > /usr/local/bin/clang-tidy-sycl <<'BASH'
#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Analyze both SYCL compilation modes and fail if either pass reports an error.
set -u
compiler=/opt/intel/oneapi/compiler/2024.2/bin/compiler
database=
for arg do
    case "$arg" in
        --) break;;
        -p=*) database=${arg#-p=};;
    esac
done
sycl=false
host_args=()
for arg do
    case "$arg" in
        --version) exec "$compiler/clang" -cc1 -version;;
        --list-checks|-list-checks|--dump-config|-dump-config|--explain-config|-explain-config|--verify-config|-verify-config|--help|-help|-h|--help-hidden)
            exec "$compiler/clang-tidy" "$@";;
        -fsycl)
            sycl=true
            host_args+=(-fno-sycl -Xclang -fsycl-is-host -D__SYCL_ANY_DEVICE_HAS_ANY_ASPECT__=1
                -isystem /opt/intel/oneapi/compiler/2024.2/include/sycl
                -isystem /opt/intel/oneapi/compiler/2024.2/include/sycl/stl_wrappers
                -isystem /opt/intel/oneapi/compiler/2024.2/include
                -isystem /opt/intel/oneapi/compiler/2024.2/opt/compiler/include)
            ;;
        *) host_args+=("$arg");;
    esac
done
if [[ -n "$database" ]]; then
    # Keep the same compiler arguments and working directory as the CMake invocation.
    exec python3 - "$0" "$database" "$@" <<'PYTHON'
import json
import os
import shlex
import subprocess
import sys

adapter, database, *args = sys.argv[1:]
source = args.pop()
options = [arg for arg in args if arg != "-p=" + database]
with open(os.path.join(database, "compile_commands.json")) as stream:
    commands = json.load(stream)
commands = [command for command in commands if os.path.abspath(os.path.join(command["directory"], command["file"])) == os.path.abspath(source)]
if not commands:
    sys.exit("No compilation command for " + source)
status = 0
for command in commands:
    argv = command.get("arguments") or shlex.split(command["command"])
    result = subprocess.call([adapter, *command.get("tidy_args", []), *options, source, "--", *argv], cwd=command["directory"])
    status = result or status
sys.exit(status)
PYTHON
fi
case "${0##*/}" in
    clang-tidy-sycl-host) exec "$compiler/clang-tidy" "${host_args[@]}";;
    clang-tidy-sycl-device)
        "$sycl" || exit 0
        exec "$compiler/clang-tidy" "$@"
        ;;
esac
status=0
"$compiler/clang-tidy" "$@" || status=$?
if "$sycl"; then
    "$compiler/clang-tidy" "${host_args[@]}" || status=$?
fi
exit "$status"
BASH
chmod +x /usr/local/bin/clang-tidy-sycl
ln -s clang-tidy-sycl /usr/local/bin/clang-tidy-sycl-host
ln -s clang-tidy-sycl /usr/local/bin/clang-tidy-sycl-device
SETUP

# Install build dependencies
ADD install_build_dependencies.sh /install_build_dependencies.sh
RUN chmod +x /install_build_dependencies.sh && \
    bash -e /install_build_dependencies.sh && \
    rm -rf /var/lib/apt/lists/*

# Install Intel offline compiler (OCLOC)
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
        gpg --yes --dearmor \
            --output /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble/lts/2523 unified" \
        > /etc/apt/sources.list.d/intel-gpu-noble.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        intel-ocloc \
        intel-ocloc-dev \
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

# Use Python 3.11 as default
# Using venv here 'cause other methods to switch the default Python on Ubuntu break both system and wheels build
RUN python3.11 -m venv venv
ENV PATH="/venv/bin:$SCCACHE_HOME:$PATH"

# Setup pip
ENV PIP_VERSION="24.0"
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.11 get-pip.py --no-cache-dir pip==${PIP_VERSION} && \
    rm -f get-pip.py

# Install newer ninja (apt ships 1.10.1 on 22.04 which lacks --quiet)
RUN python3 -m pip install --no-cache-dir ninja==1.13.0

ENV PIP_CACHE_DIR=/mount/caches/pip/linux/${PIP_VERSION}

# OneAPI env
ENV ONEAPI_ROOT=/opt/intel/oneapi
ENV PKG_CONFIG_PATH=/opt/intel/oneapi/compiler/2024.2/lib/pkgconfig
ENV DIAGUTIL_PATH=/opt/intel/oneapi/dpcpp-ct/2024.2/etc/dpct/sys_check/sys_check.sh:/opt/intel/oneapi/debugger/2024.2/etc/debugger/sys_check/sys_check.py:/opt/intel/oneapi/compiler/2024.2/etc/compiler/sys_check/sys_check.sh
ENV MANPATH=/opt/intel/oneapi/debugger/2024.2/share/man:/opt/intel/oneapi/compiler/2024.2/share/man:
ENV GDB_INFO=/opt/intel/oneapi/debugger/2024.2/share/info/
ENV CMAKE_PREFIX_PATH=/opt/intel/oneapi/compiler/2024.2
ENV CMPLR_ROOT=/opt/intel/oneapi/compiler/2024.2
ENV INFOPATH=/opt/intel/oneapi/debugger/2024.2/share/info
ENV LIBRARY_PATH=/opt/intel/oneapi/compiler/2024.2/lib
ENV OCL_ICD_FILENAMES=/opt/intel/oneapi/compiler/2024.2/lib/libintelocl.so
ENV LD_LIBRARY_PATH=/opt/intel/oneapi/debugger/2024.2/opt/debugger/lib:/opt/intel/oneapi/compiler/2024.2/opt/compiler/lib:/opt/intel/oneapi/compiler/2024.2/lib
ENV NLSPATH=/opt/intel/oneapi/mkl/2024.2/share/locale/%l_%t/%N:/opt/intel/oneapi/compiler/2024.2/lib/compiler/locale/%l_%t/%N
ENV PATH=$PATH:/opt/intel/oneapi/dev-utilities/2024.2/bin:/opt/intel/oneapi/debugger/2024.2/opt/debugger/bin:/opt/intel/oneapi/compiler/2024.2/bin
ENV INTEL_PYTHONHOME=/opt/intel/oneapi/debugger/2024.2/opt/debugger
ENV CPATH=/opt/intel/oneapi/dpl/2022.6/include:/opt/intel/oneapi/dev-utilities/2024.2/include:

# Set Intel DPC++ as a default compiler
ENV CC=icx
ENV CXX=icpx
