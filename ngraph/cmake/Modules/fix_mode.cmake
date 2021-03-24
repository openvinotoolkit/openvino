# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(MODE_APPLY_FILE PATH)
    execute_process(COMMAND git update-index --add --chmod=-x ${PATH}
        OUTPUT_VARIABLE RESULT
        ERROR_QUIET)
endfunction()

set(DIRECTORIES_OF_INTEREST
    src
    doc
    test
    python/pyngraph
)

foreach(DIRECTORY ${DIRECTORIES_OF_INTEREST})
    set(DIR "${NGRAPH_SOURCE_DIR}/${DIRECTORY}/*.?pp")
    file(GLOB_RECURSE XPP_FILES ${DIR})
    foreach(FILE ${XPP_FILES})
        mode_apply_file(${FILE})
    endforeach(FILE)
endforeach(DIRECTORY)
