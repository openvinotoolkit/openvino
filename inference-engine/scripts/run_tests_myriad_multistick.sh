#!/bin/bash
# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

APP_NAME="MyriadFunctionalTests"
APPS_TO_RUN=$1
APPS_TO_RUN=${APPS_TO_RUN:=4}

echo "Run in parallel ${APPS_TO_RUN} applications"

TEST_DIR=../../bin/intel64

# Path to test dir is provided
if [[ -n "$2" ]]; then
    TEST_DIR=$2
# Search for test dir with binaries
else
    # Windows default
    if [[ -f "${TEST_DIR}/${APP_NAME}" ]]; then
        TEST_DIR=${TEST_DIR}
    # Search for Release or Debug config
    elif [[ -f "${TEST_DIR}/Release/${APP_NAME}" ]]; then
        TEST_DIR="$TEST_DIR/Release/"
    elif [[ -f "${TEST_DIR}/Debug/${APP_NAME}" ]]; then
        TEST_DIR="$TEST_DIR/Debug/"
    else
        echo "Directory with binaries not found!"
        exit -1
    fi

fi

echo "Test directory: ${TEST_DIR}"
cd ${TEST_DIR}

export IE_VPU_MYRIADX=1

pids=""

if [[ "${APPS_TO_RUN}" -ge 1 ]] ; then
    ./${APP_NAME} --gtest_filter=*VPURegTest*SSD*myriad* &
    pids+=" $!"
fi

if [[ "${APPS_TO_RUN}" -ge 2 ]] ; then
    ./${APP_NAME} --gtest_filter=*VPURegTest*VGG*myriad* &
    pids+=" $!"
fi

if [[ "${APPS_TO_RUN}" -ge 3 ]] ; then
    ./${APP_NAME} --gtest_filter=*VPURegTest*VGG*myriad* &
    pids+=" $!"
fi

if [[ "${APPS_TO_RUN}" -ge 4 ]] ; then
    # For more then 4 multidevice testing
    for (( VAR = 4; VAR <= ${APPS_TO_RUN}; ++VAR )); do
        ./${APP_NAME} --gtest_filter=*VPURegTest*YOLO*myriad* &
        pids+=" $!"
    done
fi


# Wait for all processes to finish
sts=""
for p in ${pids}; do
    if wait ${p}; then
        sts+=" 1"
    else
        sts+=" 0"
    fi
    echo "--- Process $p finished"
done

idx=0
exit_code=0
for s in ${sts}; do
    if [[ ${s} -eq 1 ]]; then
        echo "Task $idx PASSED"
    else
        echo "Task $idx FAILED"
        exit_code=1
    fi
    ((idx+=1))
done

exit ${exit_code}
