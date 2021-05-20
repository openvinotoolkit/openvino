# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME gnaFuncTests)

addIeTargetTest(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDENCIES
            GNAPlugin
        LINK_LIBRARIES
            funcSharedTests
        ADD_CPPLINT
        LABELS
            GNA
)

# CVS-55347
set_source_files_properties(
    "${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/add_output.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shared_tests_instances/behavior/memory_states.cpp"
    PROPERTIES INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:inference_engine_legacy,INTERFACE_INCLUDE_DIRECTORIES>)

