# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(EXISTS "${REPORT_DIR}")
    file(REMOVE_RECURSE "${REPORT_DIR}")
endif()

file(MAKE_DIRECTORY "${REPORT_DIR}")

execute_process(
    COMMAND
        "${Python3_EXECUTABLE}"
        "${CONVERT_SCRIPT}"
        "--file=${INPUT_FILE}"
        "--report-dir=${REPORT_DIR}"
        "--source-dir=${SOURCE_DIR}"
        "--title=${TITLE}")

# Change cppcheck things to cpplint

file(READ "${REPORT_DIR}/index.html" cur_file_content)

string(REPLACE "Cppcheck" "cpplint" cur_file_content "${cur_file_content}")
string(REPLACE "a tool for static C/C++ code analysis" "an open source lint-like tool from Google" cur_file_content "${cur_file_content}")
string(REPLACE "http://cppcheck.sourceforge.net" "http://google-styleguide.googlecode.com/svn/trunk/cpplint/cpplint.py" cur_file_content "${cur_file_content}")
string(REPLACE "IRC: <a href=\"irc://irc.freenode.net/cppcheck\">irc://irc.freenode.net/cppcheck</a>" " " cur_file_content "${cur_file_content}")

file(WRITE "${REPORT_DIR}/index.html" "${cur_file_content}")
