#!/bin/bash

# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -e

# default ONNX Model Zoo commit hash ID:
ONNX_SHA=5faef4c33eba0395177850e1e31c4a6a9e634c82

MODELS_DIR="$HOME/.onnx/model_zoo"
ENABLE_ONNX_MODELS_ZOO=false
ENABLE_MSFT_MODELS=false
FORCE_MODE=false

function print_help {
    echo "Model preprocessing options:"
    echo "    -h display this help message"
    echo "    -d <DIR> set location of the models (for onnx model ZOO and MSFT models)"
    echo "    By default the models location is: $HOME/.onnx/model_zoo"
    echo "    -o update Onnx Model Zoo models"
    echo "    -s Onnx Model Zoo commit SHA"
    echo "    -m update MSFT models"
    echo "    -f force update of a chosen model"
    echo ""
    echo "Note: This script requires wget, GNU tar (not bsdtar) and git with LFS support."
}

while getopts "homfd:s:" opt; do
    case ${opt} in
        h )
            print_help
            ;;
        \? )
            print_help
            ;;
        : )
            print_help
            ;;
        d )
            MODELS_DIR="$OPTARG"
            ;;
        o )
            ENABLE_ONNX_MODELS_ZOO=true
            ;;
        s )
            ONNX_SHA="$OPTARG"
            ;;
        m )
            ENABLE_MSFT_MODELS=true
            ;;
        f )
            FORCE_MODE=true
            ;;
    esac
done
shift $((OPTIND -1))

MODEL_ZOO_DIR="$MODELS_DIR"
ONNX_MODELS_DIR="$MODEL_ZOO_DIR/onnx_model_zoo"
MSFT_MODELS_DIR="$MODEL_ZOO_DIR/MSFT"

function pull_and_postprocess_onnx_model_zoo() {
    git fetch
    git reset HEAD --hard

    git checkout -f "$ONNX_SHA"

    echo "Pulling models data via Git LFS for onnx model zoo repository"
    git lfs pull --include="*" --exclude="*.onnx"
    find "$ONNX_MODELS_DIR" -name "*.onnx" | while read -r filename; do rm "$filename"; done;

    printf "Extracting tar.gz archives into %s\n" "$ONNX_MODELS_DIR"
    find "$ONNX_MODELS_DIR" -name '*.tar.gz' \
        -execdir sh -c 'BASEDIR=$(basename "$1" .tar.gz) && rm -rf $BASEDIR && mkdir -p $BASEDIR' shell {} \; \
        -execdir sh -c 'BASEDIR=$(basename "$1" .tar.gz) && tar --warning=no-unknown-keyword -xvzf "$1" -C $BASEDIR' shell {} \;

    echo "Postprocessing of ONNX Model Zoo models:"

    echo "Fix roberta model"
    cd "$ONNX_MODELS_DIR/text/machine_comprehension/roberta/model/roberta-sequence-classification-9/roberta-sequence-classification-9"
    mkdir -p test_data_set_0
    mv ./*.pb test_data_set_0/

    rm -f "$MODEL_ZOO_DIR/executing_$ONNX_SHA"
}

function update_onnx_models() {
    if test "$(find "$MODEL_ZOO_DIR/executing_$ONNX_SHA" -mmin +60 2>/dev/null)" ; then
        rm -rf "$ONNX_MODELS_DIR"
        rm -f "$MODEL_ZOO_DIR/executing_$ONNX_SHA"
    fi

    while [[ -f $MODEL_ZOO_DIR/executing_$ONNX_SHA ]];
        do
            echo "Onnx Models update are currently executing - sleeping 5 minutes"
            sleep 300
        done

    if [[ ! -d "$ONNX_MODELS_DIR" ]] ; then
        touch "$MODEL_ZOO_DIR/executing_$ONNX_SHA"
        trap 'rm -f "$MODEL_ZOO_DIR/executing_$ONNX_SHA"' EXIT INT TERM
        echo "The ONNX Model Zoo repository doesn't exist on your filesystem then will be cloned"
        git clone https://github.com/onnx/models.git "$ONNX_MODELS_DIR"
        cd "$ONNX_MODELS_DIR"
        pull_and_postprocess_onnx_model_zoo
    else
        # Check if ONNX Model Zoo directory consists of proper git repo
        git_remote_url=$(git -C "$ONNX_MODELS_DIR" config --local remote.origin.url 2> /dev/null 2>&1)
        printf "ONNX Model Zoo repository exists: %s\n" "$ONNX_MODELS_DIR"
        if [[ $git_remote_url = "https://github.com/onnx/models.git" ]]; then
            printf "The proper github repository detected: %s\n" "$git_remote_url"
        else
            echo "The ONNX Model Zoo repository doesn't exist then will be cloned"
            git clone https://github.com/onnx/models.git "$ONNX_MODELS_DIR"
        fi
    fi
}

function update_msft_models() {
    wget https://onnxruntimetestdata.blob.core.windows.net/models/20191107.zip -O "$MSFT_MODELS_DIR.zip"
    unzip "$MSFT_MODELS_DIR.zip" -d "$MSFT_MODELS_DIR" && rm "$MSFT_MODELS_DIR.zip"

}

function postprocess_msft_models() {
    echo "Postprocessing of MSFT models:"

    echo "Fix LSTM_Seq_lens_unpacked"
    mv "$MSFT_MODELS_DIR"/opset9/LSTM_Seq_lens_unpacked/seq_lens_sorted "$MSFT_MODELS_DIR"/opset9/LSTM_Seq_lens_unpacked/test_data_set_0
    mv "$MSFT_MODELS_DIR"/opset9/LSTM_Seq_lens_unpacked/seq_lens_unsorted "$MSFT_MODELS_DIR"/opset9/LSTM_Seq_lens_unpacked/test_data_set_1
}

if [[ $ENABLE_ONNX_MODELS_ZOO = false ]] && [[ $ENABLE_MSFT_MODELS = false ]] ; then
    echo "Please choose an option to update chosen model:
            -o to update ONNX Model ZOO
            -m to update MSFT models"
    exit 170
fi

if [[ $MODELS_DIR = false ]] ; then
    printf "Unknown location of the general models directory (onnx model ZOO and MSFT models)
            Please specify the location using -d <DIR> flag"
    exit 170
fi


# check if general model zoo directory exists (directory to store ONNX model zoo and MSFT models)
if [[ ! -d "$MODEL_ZOO_DIR" ]] ; then
    printf "The general model directory: %s doesn't exist on your filesystem, it will be created \n" "$MODEL_ZOO_DIR"
    mkdir -p "$MODEL_ZOO_DIR"
else
    printf "The general model directory: %s found\n" "$MODEL_ZOO_DIR"
fi

if [[ $ENABLE_ONNX_MODELS_ZOO = true ]] ; then
    if [[ $FORCE_MODE = true ]]; then
        rm -rf "$ONNX_MODELS_DIR"
    fi
    update_onnx_models
fi

if [[ $ENABLE_MSFT_MODELS = true ]] ; then
    if [[ $FORCE_MODE = true ]]; then
        rm -rf "$MSFT_MODELS_DIR"
    fi
    update_msft_models
    postprocess_msft_models
fi
