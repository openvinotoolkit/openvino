# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(WRITE "${FINAL_OUTPUT_FILE}" "")

foreach(output_file IN LISTS OUTPUT_FILES)
    file(READ "${output_file}" cur_file_content)
    file(APPEND "${FINAL_OUTPUT_FILE}" "${cur_file_content}\n")
endforeach()
