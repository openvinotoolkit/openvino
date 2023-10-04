#!/usr/bin/bash -e

PREV=0
TEST=ov_gpu_unit_tests
FILE=mha_opt.cl
build_and_measure() {
    TITLE=$1
    # ninja -C ../../../build -j 24 ov_gpu_unit_tests | grep -v processing &> /dev/null
    NS=`CLI_DevicePerformanceTiming=1 cliloader $TEST --gtest_filter=*mha_graph_test_f2_N9216_d64* 2>&1 | grep mha_opt | cut -d',' -f 3`
    US=$((NS/1000))
    echo "$TITLE: $US us, $((US-PREV)) us"
    PREV=$((US))
}
if [[ ! -f mha_opt.cl ]]; then
    echo "cannot find mha_opt.cl file"
    echo "Please execute this script within cl_kernel directory"
    exit 1
fi

# Prepare
sed -i -e 's/define MEASURE_BLOCK_/define AMEASURE_BLOCK_/' $FILE
sed -i -e 's/define RETURN_BLOCK_/define ARETURN_BLOCK_/' $FILE
sed -i -e 's/define MEASURE$/define AMEASURE/' $FILE

# Stage
sed -i -e 's/define AMEASURE_BLOCK_1/define MEASURE_BLOCK_1/' $FILE
sed -i -e 's/define .*RETURN_BLOCK_.*/define RETURN_BLOCK_1/' $FILE
sed -i -e 's/define AMEASURE$/define MEASURE/' $FILE

build_and_measure "STAGE_1"

# Stage
sed -i -e 's/define AMEASURE_BLOCK_2/define MEASURE_BLOCK_2/' $FILE
sed -i -e 's/define .*RETURN_BLOCK_.*/define RETURN_BLOCK_2/' $FILE
sed -i -e 's/define AMEASURE$/define MEASURE/' $FILE

build_and_measure "STAGE_2"

# Stage
sed -i -e 's/define AMEASURE_BLOCK_3/define MEASURE_BLOCK_3/' $FILE
sed -i -e 's/define .*RETURN_BLOCK_.*/define RETURN_BLOCK_3/' $FILE
sed -i -e 's/define AMEASURE$/define MEASURE/' $FILE

build_and_measure "STAGE_3"

# Stage
sed -i -e 's/define AMEASURE_BLOCK_4/define MEASURE_BLOCK_4/' $FILE
sed -i -e 's/define .*RETURN_BLOCK_.*/define RETURN_BLOCK_4/' $FILE
sed -i -e 's/define AMEASURE$/define MEASURE/' $FILE

build_and_measure "STAGE_4"

# Stage
sed -i -e 's/define AMEASURE_BLOCK_5/define MEASURE_BLOCK_5/' $FILE
sed -i -e 's/define .*RETURN_BLOCK_.*/define RETURN_BLOCK_5/' $FILE
sed -i -e 's/define AMEASURE$/define MEASURE/' $FILE

build_and_measure "STAGE_5"

# Disable profiling macros
sed -i -e 's/define MEASURE_BLOCK_/define AMEASURE_BLOCK_/' $FILE
sed -i -e 's/define RETURN_BLOCK_/define ARETURN_BLOCK_/' $FILE
sed -i -e 's/define MEASURE$/define AMEASURE/' $FILE

build_and_measure "STAGE_LAST"

