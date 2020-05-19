#!/bin/bash
set -e

echo "--TRAVIS VARIABLES--"
echo "TRAVIS_OS_NAME:" ${TRAVIS_OS_NAME}
echo "TRAVIS_BUILD_DIR:" ${TRAVIS_BUILD_DIR}

echo "--CUSTOM VARIABLES--"
echo "TASK:" ${TASK}
echo "OS:" ${OS}

# LINUX TASKS
if [ ${TRAVIS_OS_NAME} == "linux" ]; then

    if [ ${TASK} == "cpp_test" ]; then
        docker run -w '/root/ngraph/build' test_ngraph make unit-test-check
    fi

    if [ ${TASK} == "python2_test" ]; then
        docker run -w '/root/ngraph/python' -e NGRAPH_ONNX_IMPORT_ENABLE=TRUE test_ngraph tox -e py27
    fi

    if [ ${TASK} == "python3_test" ]; then
        docker run -w '/root/ngraph/python' -e NGRAPH_ONNX_IMPORT_ENABLE=TRUE test_ngraph tox -e py3
    fi

fi

# MacOS TASKS
if [ ${TRAVIS_OS_NAME} == "osx" ]; then

    if [ ${TASK} == "cpp_test" ]; then
        cd ${TRAVIS_BUILD_DIR}/build
        make unit-test-check
    fi

fi
