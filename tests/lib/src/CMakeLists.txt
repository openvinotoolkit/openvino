# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "shared_tests_lib")

file (GLOB SRC *.cpp)
add_library(${TARGET_NAME} STATIC ${SRC})

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#set(OpenVINO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../")

# Search OpenVINO Inference Engine via InferenceEngine_DIR
find_package(IEDevScripts REQUIRED
             PATHS "${OpenVINO_SOURCE_DIR}/cmake/developer_package"
             NO_CMAKE_FIND_ROOT_PATH
             NO_DEFAULT_PATH)

# Search OpenVINO Inference Engine installed
find_package(InferenceEngine)

if(NOT InferenceEngine_FOUND)
    # Search OpenVINO Inference Engine via InferenceEngineDeveloperPackage_DIR
    # in order to provide backward compatibility with old OpenVINO packages
    find_package(InferenceEngineDeveloperPackage REQUIRED)
endif()

target_link_libraries(${TARGET_NAME} PUBLIC ${InferenceEngine_LIBRARIES})
