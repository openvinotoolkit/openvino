# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(style_apply_file INPUT_FILE)
    execute_process(COMMAND ${CLANG_FORMAT} -style=file -i ${INPUT_FILE}
        OUTPUT_VARIABLE STYLE_CHECK_RESULT)
endfunction()

foreach(source_file IN LISTS INPUT_FILES)
    set(exclude FALSE)
    foreach(pattern IN LISTS EXCLUDE_PATTERNS)
        if(source_file MATCHES "${pattern}")
            set(exclude ON)
            break()
        endif()
    endforeach()

    if(exclude)
        continue()
    endif()

    style_apply_file(${source_file})
endforeach()
