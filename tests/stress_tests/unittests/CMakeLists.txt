# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "StressUnitTests")

file (GLOB SRC
        ../common/*.cpp
        ../common/ie_pipelines/*.cpp
        *.cpp
        tests_pipelines/*.cpp)

file (GLOB HDR
        ../common/*.h
        ../common/managers/*.h
        ../common/ie_pipelines/*.h
        *.h
        tests_pipelines/*.h)

# Create library file from sources.
add_executable(${TARGET_NAME} ${HDR} ${SRC})

target_link_libraries(${TARGET_NAME}
        IE::gtest
        IE::gtest_main
        IE::pugixml
        gflags
        Threads::Threads
        ${InferenceEngine_LIBRARIES}
        )

target_include_directories(${TARGET_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}"
        "${IE_MAIN_SOURCE_DIR}/thirdparty/pugixml/src")

# Copy local configs to BIN_FOLDER
configure_file(local_configs/test_config.xml ${OUTPUT_ROOT}/${BIN_FOLDER}/${CMAKE_BUILD_TYPE}/stress_tests_configs/unittests/test_config.xml COPYONLY)
