#!/bin/bash
set -e

MODELS_DIR=false
CLEAN_DIR=false
ENABLE_MSFT=false
CLONE=false

function print_help {
    echo "Model preprocessing options:"
    echo "    -h display this help message"
    echo "    -c clone ONNX models repository"
    echo "    -m <DIR> set location of the models"
    echo "    -f clean target directory(during clone or after enable MSFT models)"
    echo "    -e download and prepare MSFT models"
}

while getopts ":hcefm:" opt; do
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
        c )
            CLONE=true
            ;;
        m )
            MODELS_DIR="$OPTARG"
            ;;
        f )
            CLEAN_DIR=true
            ;;
        e )
            ENABLE_MSFT=true
            ;;
    esac
done
shift $((OPTIND -1))

if [ "$MODELS_DIR" = false ] ; then
    echo "Unknown location of the ZOO models"
    exit 170
fi

ONNX_MODELS_DIR="$MODELS_DIR/model_zoo/onnx_model_zoo"
MSFT_MODELS_DIR="$MODELS_DIR/model_zoo/MSFT"

if [ "$CLONE" = true ] ; then
    if [ "$CLEAN_DIR" = true ] ; then
        rm -rf "$ONNX_MODELS_DIR"
    fi
    git clone https://github.com/onnx/models.git "$ONNX_MODELS_DIR"
fi

mkdir -p "$ONNX_MODELS_DIR"
cd "$ONNX_MODELS_DIR"
# remove already downloaded models
git clean -f -x -d
git checkout .
git pull -p
# pull models from the lfs repository
# onnx models are included in the tar.gz archives
git lfs pull --include="*" --exclude="*.onnx"
find "$ONNX_MODELS_DIR" -name "*.onnx" | while read filename; do rm "$filename"; done;
echo "extracting tar.gz archives..."
find "$ONNX_MODELS_DIR" -name '*.tar.gz' -execdir sh -c 'BASEDIR=$(basename "{}" .tar.gz) && mkdir -p $BASEDIR' \; -execdir sh -c 'BASEDIR=$(basename "{}" .tar.gz) && tar -xzvf "{}" -C $BASEDIR' \;
# fix yolo v4 model
cd "$ONNX_MODELS_DIR/vision/object_detection_segmentation/yolov4/model/yolov4/yolov4/test_data_set"
mv input0.pb input_0.pb
mv input1.pb input_1.pb
mv input2.pb input_2.pb
mv output0.pb output_0.pb
mv output1.pb output_1.pb
mv output2.pb output_2.pb
# fix roberta model
cd "$ONNX_MODELS_DIR/text/machine_comprehension/roberta/model/roberta-sequence-classification-9/roberta-sequence-classification-9"
mkdir test_data_set_0
mv *.pb test_data_set_0/

# Prepare MSFT models
if [ "$ENABLE_MSFT" = true ] ; then
    if [ "$CLEAN_DIR" = true ] ; then
        rm -rf "$MSFT_MODELS_DIR"
    fi
    mkdir -p "$MSFT_MODELS_DIR"
    wget -O "$MSFT_MODELS_DIR/20191107.zip" https://onnxruntimetestdata.blob.core.windows.net/models/20191107.zip
    unzip "$MSFT_MODELS_DIR/20191107.zip" -d "$MSFT_MODELS_DIR/20191107" && rm "$MSFT_MODELS_DIR/20191107.zip"
fi
