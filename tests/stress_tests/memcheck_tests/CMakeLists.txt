# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "MemCheckTests")

file (GLOB_RECURSE SRC *.cpp)
file (GLOB_RECURSE HDR *.h)

# Create library file from sources.
add_executable(${TARGET_NAME} ${HDR} ${SRC})

target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE StressTestsCommon)

install(TARGETS ${TARGET_NAME}
            RUNTIME DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)

# Copy local configs to BIN_FOLDER
configure_file(local_configs/test_config.xml 
        ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/stress_tests_configs/memcheck_tests/test_config.xml COPYONLY)
configure_file(local_configs/references_config.xml
        ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/stress_tests_configs/memcheck_tests/references_config.xml COPYONLY)
