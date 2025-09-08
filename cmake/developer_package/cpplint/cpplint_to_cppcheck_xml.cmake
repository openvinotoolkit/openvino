# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

execute_process(
    COMMAND
        "${Python3_EXECUTABLE}"
        "${CONVERT_SCRIPT}"
    INPUT_FILE "${INPUT_FILE}"
    OUTPUT_FILE "${OUTPUT_FILE}"
    ERROR_FILE "${OUTPUT_FILE}")
