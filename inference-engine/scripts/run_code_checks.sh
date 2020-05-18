#!/bin/bash
# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
command -v realpath >/dev/null 2>&1 || { echo >&2 "cpplint require realpath executable but it's not installed.  Aborting."; exit 1; }
SOURCE_DIR=$(realpath ${CURRENT_DIR}/..)
REPORT_DIR="${SOURCE_DIR}/report"
CPPLINT_REPORT_DIR="${REPORT_DIR}/cpplint"
PROJECT_NAME="Inference Engine"

function run_cpplint() {
    echo "-> CppLint started..."
    if [ -d ${CPPLINT_REPORT_DIR} ]; then
        rm -Rf ${CPPLINT_REPORT_DIR}
    fi

    mkdir -p ${CPPLINT_REPORT_DIR}
    python ${CURRENT_DIR}/cpplint.py --linelength=160 --counting=detailed --quiet --filter="
        -build/header_guard,
        -build/include,
        -build/include_order,
        -build/include_subdir,
        -build/include_what_you_use,
        -build/namespaces,
        -build/c++11,
        -whitespace/indent,
        -whitespace/comments,
        -whitespace/ending_newline,
        -runtime/references,
        -runtime/int,
        -runtime/explicit,
        -readability/todo,
        -readability/fn_size
    " $(find ${SOURCE_DIR} -name '*.h' -or -name '*.cc' -or -name '*.c' -or -name '*.cpp' -or -name '*.hpp' |
        grep -v 'inference-engine/bin\|inference-engine/build\|inference-engine/report\|inference-engine/scripts\|inference-engine/temp\|inference-engine/tests_deprecated/\|gtest\|inference-engine/ie_bridges\|pugixml\|inference-engine/tools/vpu_perfcheck\|thirdparty/gflags\|thirdparty/ade\|thirdparty/fluid\|thirdparty/mkl-dnn\|thirdparty/movidius\|thirdparty/ocv\|thirdparty/plugixml\|thirdparty/std_lib\|thirdparty/clDNN/common\|thirdparty/clDNN/tutorial\|thirdparty/clDNN/utils' |
        grep 'include\|src\|inference-engine/samples\|thirdparty/clDNN/kernel_selector\|thirdparty/clDNN/api\|thirdparty/clDNN/api_extension\|inference-engine/tests_' ) 2>&1 |
        sed 's/"/\&quot;/g' >&1| sed 's/</\&lt;/g' >&1| sed 's/>/\&gt;/g' >&1| sed "s/'/\&apos;/g" >&1|
        sed 's/\&/\&amp;/g' >&1| python ${CURRENT_DIR}/cpplint_to_cppcheckxml.py &> ${CPPLINT_REPORT_DIR}/cpplint-cppcheck-result.xml

	# Generate html from it
	${CURRENT_DIR}/cppcheck-htmlreport.py --file=${CPPLINT_REPORT_DIR}/cpplint-cppcheck-result.xml --report-dir=${CPPLINT_REPORT_DIR} --source-dir=${SOURCE_DIR} --title=${PROJECT_NAME}

	# Change Cppcheck things to cpplint
	sed -i.bak 's/Cppcheck/cpplint/g' ${CPPLINT_REPORT_DIR}/index.html
	sed -i.bak 's/a\ tool\ for\ static\ C\/C++\ code\ analysis/an\ open\ source\ lint\-like\ tool\ from\ Google/g' ${CPPLINT_REPORT_DIR}/index.html
	sed -i.bak 's/http:\/\/cppcheck.sourceforge.net/http:\/\/google\-styleguide.googlecode.com\/svn\/trunk\/cpplint\/cpplint.py/g' ${CPPLINT_REPORT_DIR}/index.html
	sed -i.bak 's/IRC: <a href=\"irc:\/\/irc.freenode.net\/cppcheck\">irc:\/\/irc.freenode.net\/cppcheck<\/a>/\ /g' ${CPPLINT_REPORT_DIR}/index.html

    echo "-> CppLint finished..."
}

function run_cpp_check() {
    echo "-> Cppcheck started..."
    CPPCHECK_REPORT_DIR="${REPORT_DIR}/cppcheck"
    if [ -d ${CPPCHECK_REPORT_DIR} ]; then
        rm -Rf ${CPPCHECK_REPORT_DIR}
    fi

    mkdir -p ${CPPCHECK_REPORT_DIR}

	# Generate cppcheck xml
	cppcheck -v --enable=all --suppress=missingIncludeSystem --std=c++11 ${SOURCE_DIR} -i${SOURCE_DIR}/thirdparty -i${SOURCE_DIR}/tests/libs -i${SOURCE_DIR}/temp -i${SOURCE_DIR}/build \
	  -i${SOURCE_DIR}/bin -i${SOURCE_DIR}/report -I${SOURCE_DIR}/include -I${SOURCE_DIR}/src -I${SOURCE_DIR}/thirdparty/pugixml/src -I${SOURCE_DIR}/thirdparty/gflags/src -I${SOURCE_DIR}/samples/scoring_agent/HTTPClient -I${SOURCE_DIR}/src/inference_engine --xml-version=2 2> ${CPPCHECK_REPORT_DIR}/cppcheck-only-result.xml

	# Generate html from it
	python ${CURRENT_DIR}/cppcheck-htmlreport.py\
		--file=${CPPCHECK_REPORT_DIR}/cppcheck-only-result.xml\
		--report-dir=${CPPCHECK_REPORT_DIR}\
		--source-dir=${SOURCE_DIR}\
		--title=${PROJECT_NAME}
    echo "-> Cppcheck finished..."
}

if [ ! -d ${REPORT_DIR} ]; then
    mkdir -p ${REPORT_DIR}
fi

run_cpplint

out_cpp_lint=`cat ${CPPLINT_REPORT_DIR}/cpplint-cppcheck-result.xml`
if [[ ${out_cpp_lint} == *"error"* ]]; then
    exit 1
fi
#run_cpp_check
