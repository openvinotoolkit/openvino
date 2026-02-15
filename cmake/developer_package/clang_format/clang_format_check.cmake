# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file(REMOVE "${OUTPUT_FILE}")

execute_process(COMMAND ${CLANG_FORMAT} -style=file -output-replacements-xml ${INPUT_FILE}
    OUTPUT_VARIABLE STYLE_CHECK_RESULT
    )

file(WRITE "${OUTPUT_FILE}" "${STYLE_CHECK_RESULT}")

if(NOT SKIP_RETURN_CODE)
    if("${STYLE_CHECK_RESULT}" MATCHES ".*<replacement .*")
        message(FATAL_ERROR "[clang-format] Code style check failed for: ${INPUT_FILE}")
    endif()
endif()
