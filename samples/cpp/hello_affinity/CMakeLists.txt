# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(POLICY CMP0079)
    cmake_policy(SET CMP0079 NEW)
endif()

if(NOT TARGET nlohmann_json::nlohmann_json)
    if(EXISTS "${Samples_SOURCE_DIR}/thirdparty/nlohmann_json")
        # OpenVINO package puts thirdparty to samples dir
        # suppress shadowing names warning
        set(JSON_SystemInclude ON CACHE BOOL "" FORCE)
        add_subdirectory("${Samples_SOURCE_DIR}/thirdparty/nlohmann_json"
                         "${Samples_BINARY_DIR}/thirdparty/nlohmann_json" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${Samples_SOURCE_DIR}/../../thirdparty/json/nlohmann_json")
        # Allow running samples CMakeLists.txt as stand alone from openvino sources
        # suppress shadowing names warning
        set(JSON_SystemInclude ON CACHE BOOL "" FORCE)
        add_subdirectory("${Samples_SOURCE_DIR}/../../thirdparty/json/nlohmann_json"
                         "${Samples_BINARY_DIR}/thirdparty/nlohmann_json" EXCLUDE_FROM_ALL)
    else()
        message(FATAL_ERROR "Failed to find / build nlohmann_json library")
    endif()
endif()

target_link_libraries(ie_samples_utils PRIVATE nlohmann_json::nlohmann_json)

ov_add_sample(NAME hello_affinity
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/inference.cpp"
                      "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
                      "${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/inference.hpp"
                      "${CMAKE_CURRENT_SOURCE_DIR}/utils.hpp"
              DEPENDENCIES ie_samples_utils)
