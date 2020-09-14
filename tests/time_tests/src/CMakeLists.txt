# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# add dummy `time_tests` target combines all time tests
add_custom_target(time_tests)

# Build test from every source file matchs *-pipeline.cpp.
# Test target name is source file name without extension.
FILE(GLOB tests "*-pipeline.cpp")

foreach(test_source ${tests})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})

    target_link_libraries(${test_name} PRIVATE IE::inference_engine time-testhelper)

    add_dependencies(time_tests ${test_name})
endforeach()
