#!/usr/bin/env bash

# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

echo -ne "\e[0;33mWARNING: If you get an error when running the sample in the Docker container, you may need to install additional packages. To do this, run the container as root (-u 0) and run install_openvino_dependencies.sh script. If you get a package-independent error, try setting additional parameters using -sample-options.\e[0m\n"

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]-$0}" )" && pwd )"
build_dir="$HOME/inference_engine_cpp_samples_build"

. "$ROOT_DIR/utils.sh"

usage() {
    echo "Benchmark demo using public SqueezeNet topology"
    echo
    echo "Options:"
    echo "  -help                     Print help message"
    echo "  -b BUILD_DIR              Specify the sample build directory"
    echo "  -d DEVICE                 Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD are acceptable. Sample will look for a suitable plugin for device specified"
    echo "  -sample-options OPTIONS   Specify command line arguments for the sample"
    echo
    exit 1
}

trap 'error ${LINENO}' ERR

target="CPU"

# parse command line options
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -b | --build_dir)
    build_dir="$2/inference_engine_cpp_samples_build"
    shift
    ;;
    -h | -help | --help)
    usage
    ;;
    -d)
    target="$2"
    echo target = "${target}"
    shift
    ;;
    -sample-options)
    sampleoptions=("${@:2}")
    echo sample-options = "${sampleoptions[*]}"
    shift
    ;;
    *)
    # unknown option
    ;;
esac
shift
done

if [ -z "${sampleoptions[*]}" ]; then
    sampleoptions=( -niter 1000 )
fi

target_precision="FP16"

echo -ne "target_precision = ${target_precision}\n"

models_path="$build_dir/../openvino_models/models"
models_cache="$build_dir/../openvino_models/cache"
irs_path="$build_dir/../openvino_models/ir"

model_name="squeezenet1.1"

target_image_path="$ROOT_DIR/car.png"

run_again="Then run the script again.\n\n"
omz_tool_error_message="It is required to download and convert a model. Check https://pypi.org/project/openvino-dev/ to install it. ${run_again}"

if [ -e "$ROOT_DIR/../../setupvars.sh" ]; then
    setupvars_path="$ROOT_DIR/../../setupvars.sh"
else
    echo -ne "Error: setupvars.sh is not found\n"
fi

if ! . "$setupvars_path" ; then
    echo -ne "Unable to run ./setupvars.sh. Please check its presence. ${run_again}"
    exit 1
fi

if [[ -f /etc/centos-release ]]; then
    DISTRO="centos"
elif [[ -f /etc/lsb-release ]]; then
    DISTRO="ubuntu"
elif [[ -f /etc/redhat-release ]]; then
    DISTRO="redhat"
fi

if [[ $DISTRO == "centos" ]]; then
    # check installed Python version
    if command -v python3.6 >/dev/null 2>&1; then
        python_binary=python3.6
    fi
elif [[ $DISTRO == "redhat" ]]; then
    python_binary=python3
elif [[ $DISTRO == "ubuntu" ]]; then
    python_binary=python3
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # check installed Python version
    if command -v python3.8 >/dev/null 2>&1; then
        python_binary=python3.8
    elif command -v python3.7 >/dev/null 2>&1; then
        python_binary=python3.7
    elif command -v python3.6 >/dev/null 2>&1; then
        python_binary=python3.6
    else
        python_binary=python3
    fi
fi

if ! command -v $python_binary &>/dev/null; then
    echo -ne "\n\nPython 3.6 (x64) or higher is not installed. It is required to run Model Optimizer, please install it. ${run_again}"
    exit 1
fi

if ! command -v omz_info_dumper &>/dev/null; then
    echo -ne "\n\nomz_info_dumper was not found. ${omz_tool_error_message}"
    exit 2
fi

if ! command -v omz_downloader &>/dev/null; then
    echo -ne "\n\nomz_downloader was not found. ${omz_tool_error_message}"
    exit 3
fi

if ! command -v omz_converter &>/dev/null; then
    echo -ne "\n\nomz_converter was not found. ${omz_tool_error_message}"
    exit 4
fi

# Step 1. Download the Caffe model and the prototxt of the model
echo -ne "\n###############|| Downloading the Caffe model and the prototxt ||###############\n\n"

model_dir=$(omz_info_dumper --name "$model_name" |
    ${python_binary} -c 'import sys, json; print(json.load(sys.stdin)[0]["subdirectory"])')

print_and_run omz_downloader --name "$model_name" --output_dir "${models_path}" --cache_dir "${models_cache}"

ir_dir="${irs_path}/${model_dir}/${target_precision}"

if [ ! -e "$ir_dir" ]; then
    # Step 2. Convert a model with Model Optimizer
    echo -ne "\n###############|| Convert a model with Model Optimizer ||###############\n\n"

    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
    print_and_run omz_converter --name "$model_name" -d "$models_path" -o "$irs_path" --precisions "$target_precision"
else
    echo -ne "\n\nTarget folder ${ir_dir} already exists. Skipping IR generation  with Model Optimizer."
    echo -ne "If you want to convert a model again, remove the entire ${ir_dir} folder. ${run_again}"
fi

# Step 3. Build samples
echo -ne "\n###############|| Build OpenVINO Runtime samples ||###############\n\n"

OS_PATH=$(uname -m)
NUM_THREADS="-j2"

if [ "$OS_PATH" == "x86_64" ]; then
  OS_PATH="intel64"
  NUM_THREADS="-j8"
fi

samples_path="${INTEL_OPENVINO_DIR}/samples/cpp"
binaries_dir="${build_dir}/${OS_PATH}/Release"

if [ -e "$build_dir/CMakeCache.txt" ]; then
  rm -rf "$build_dir/CMakeCache.txt"
fi
mkdir -p "$build_dir"
cd "$build_dir"
cmake -DCMAKE_BUILD_TYPE=Release "$samples_path"

make $NUM_THREADS benchmark_app

# Step 4. Run samples
echo -ne "\n###############|| Run OpenVINO Runtime benchmark app ||###############\n\n"

cd "$binaries_dir"

cp -f "$ROOT_DIR/${model_name}.labels" "${ir_dir}/"

print_and_run ./benchmark_app -d "$target" -i "$target_image_path" -m "${ir_dir}/${model_name}.xml" -pc "${sampleoptions[@]}"

echo -ne "\n###############|| OpenVINO Runtime benchmark app completed successfully ||###############\n\n"
