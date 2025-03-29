# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# add dummy `memory_tests` target combines all memory tests
add_custom_target(memory_tests)

# Build test from every source file.
# Test target name is source file name without extension.
FILE(GLOB tests "*.cpp")

foreach(test_source ${tests})
    get_filename_component(test_name ${test_source} NAME_WE)
    add_executable(${test_name} ${test_source})

    target_link_libraries(${test_name} PRIVATE tests_shared_lib memory_tests_helper)

    add_dependencies(memory_tests ${test_name})

    install(TARGETS ${test_name}
            RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
endforeach()
