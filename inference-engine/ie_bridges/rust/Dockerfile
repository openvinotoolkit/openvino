# This Dockerfile demonstrates how to build the openvino bindings using an installation of OpenVINO. For instructions
# to install OpenVINO see the OpenVINO documentation, e.g.
# https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_apt.html.
FROM rust:1.45

# Setup Rust.
RUN rustup component add rustfmt

# Install OpenVINO.
WORKDIR /tmp
RUN wget https://apt.repos.intel.com/openvino/2020/GPG-PUB-KEY-INTEL-OPENVINO-2020 && \
    echo '5f5cff8a2d26ba7de91942bd0540fa4d  GPG-PUB-KEY-INTEL-OPENVINO-2020' > CHECKSUM && \
    md5sum --check CHECKSUM && \
    apt-key add GPG-PUB-KEY-INTEL-OPENVINO-2020 && \
    echo "deb https://apt.repos.intel.com/openvino/2020 all main" | tee /etc/apt/sources.list.d/intel-openvino-2020.list && \
    apt update && \
    apt install -y intel-openvino-runtime-ubuntu18-2020.4.287

# Install build dependencies (for bindgen).
RUN apt install -y clang libclang-dev

# Copy in OpenVINO source
WORKDIR /usr/src/openvino
COPY . .

# Build openvino libraries.
WORKDIR /usr/src/openvino/inference-engine/ie_bridges/rust
RUN OPENVINO_INSTALL_DIR=/opt/intel/openvino cargo build -vv

# Test; note that we need to setup the library paths before using them since the OPENVINO_INSTALL_DIR can only affect
# the build library search path.
RUN ["/bin/bash", "-c", "source /opt/intel/openvino/bin/setupvars.sh && OPENVINO_INSTALL_DIR=/opt/intel/openvino cargo test -v"]
