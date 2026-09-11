# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(AFFINITY_UTILS_TARGET "ie_samples_affinity_utils")

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

add_library(${AFFINITY_UTILS_TARGET} STATIC EXCLUDE_FROM_ALL "${Samples_SOURCE_DIR}/common/utils/src/affinity_utils.cpp")
set_target_properties(${AFFINITY_UTILS_TARGET} PROPERTIES FOLDER "src")
target_include_directories(${AFFINITY_UTILS_TARGET}
    PUBLIC "$<BUILD_INTERFACE:${Samples_SOURCE_DIR}/common/utils/include>")
target_link_libraries(${AFFINITY_UTILS_TARGET}
    PUBLIC openvino::runtime
    PRIVATE ie_samples_utils nlohmann_json::nlohmann_json)

ov_add_sample(NAME hello_affinity
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/inference.cpp"
                      "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
                      "${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp"
              HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/inference.hpp"
                      "${CMAKE_CURRENT_SOURCE_DIR}/utils.hpp"
              DEPENDENCIES ie_samples_utils ${AFFINITY_UTILS_TARGET})
