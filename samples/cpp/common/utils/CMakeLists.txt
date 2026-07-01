# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ie_samples_utils")

file(GLOB_RECURSE SOURCES "*.cpp" "*.hpp" "*.h")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCES})

add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL ${SOURCES})
set_target_properties(${TARGET_NAME} PROPERTIES FOLDER "src")

target_include_directories(${TARGET_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

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

target_link_libraries(${TARGET_NAME} PUBLIC openvino::runtime PRIVATE ${GFLAGS_TARGET} nlohmann_json::nlohmann_json)

if(COMMAND ov_add_clang_format_target)
    ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
endif()
