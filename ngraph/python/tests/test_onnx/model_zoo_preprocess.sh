#!/bin/bash
set -e

MODELS_DIR=false

function print_help {
	echo "Model preprocessing options:"
	echo "    -h display this help message"
	echo "    -c clone ONNX models repository"
	echo "    -m <DIR> set location of the models"
	echo "    -f clean target directory(during clone)"
}

while getopts ":hcfm:" opt; do
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
			MODELS_DIR=$OPTARG
			;;
		f )
			CLEAN_DIR=true
			;;
	esac
done
shift $((OPTIND -1))

if [ $MODELS_DIR = false ] ; then
	echo "Unknown location of the ONNX ZOO models"
	exit 170
fi

if [ $CLONE = true ] ; then
	if [ $CLEAN_DIR = true ] ; then
		rm -rf $MODELS_DIR
	fi
	git clone https://github.com/onnx/models.git $MODELS_DIR
fi

cd $MODELS_DIR
# remove already downloaded models
git clean -f -x -d
git checkout .
git pull -p
# pull models from the lfs repository
# onnx models are included in the tar.gz archives
git lfs pull --include="*" --exclude="*.onnx"
find $MODELS_DIR -name "*.onnx" | while read filename; do rm "$filename"; done;
echo "extracting tar.gz archives..."
find $MODELS_DIR -name '*.tar.gz' -execdir sh -c 'BASEDIR=$(basename "{}" .tar.gz) && mkdir -p $BASEDIR' \; -execdir sh -c 'BASEDIR=$(basename "{}" .tar.gz) && tar -xzvf "{}" -C $BASEDIR --strip-components=1' \;
# fix yolo v4 model
cd $MODELS_DIR/vision/object_detection_segmentation/yolov4/model/yolov4/yolov4/test_data_set
mv input0.pb input_0.pb
mv input1.pb input_1.pb
mv input2.pb input_2.pb
mv output0.pb output_0.pb
mv output1.pb output_1.pb
mv output2.pb output_2.pb
# fix roberta model
cd $MODELS_DIR/text/machine_comprehension/roberta/model/roberta-sequence-classification-9/
mkdir test_data_set_0
mv *.pb test_data_set_0/
