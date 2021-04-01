# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# add dummy `fuzz` target combines all fuzz tests
add_custom_target(fuzz)

# Build fuzz test from every source file matchs *-fuzzer.cc.
# Fuzz test target name is source file name without extension.
FILE(GLOB tests "*-fuzzer.cc")

foreach(test_source ${tests})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_fuzzer(${test_name} ${test_source})

    target_link_libraries(${test_name} PRIVATE IE::inference_engine)

    add_dependencies(fuzz ${test_name})
endforeach()
