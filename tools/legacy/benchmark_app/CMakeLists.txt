# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME benchmark_app_legacy)

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)

source_group("src" FILES ${SRC})
source_group("include" FILES ${HDR})

add_executable(${TARGET_NAME} ${SRC} ${HDR})

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

set_target_properties(${TARGET_NAME} PROPERTIES
    COMPILE_PDB_NAME ${TARGET_NAME}
    FOLDER tools
)

if (CMAKE_COMPILER_IS_GNUCXX)
    target_compile_options(${TARGET_NAME} PRIVATE -Wall)
endif()

find_package(OpenCV COMPONENTS core QUIET)
if(NOT OpenCV_FOUND)
    message(WARNING "OpenCV is disabled or not found, ${TARGET_NAME} will be built without OpenCV support. Set OpenCV_DIR")
    target_link_libraries(${TARGET_NAME} PRIVATE nlohmann_json_schema_validator openvino::runtime format_reader gflags)
else()
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_OPENCV)
    target_link_libraries(${TARGET_NAME} PRIVATE opencv_core openvino::runtime format_reader gflags)
endif()

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

install(TARGETS ${TARGERT_NAME}
        RUNTIME DESTINATION tools/legacy/benchmark_app
        COMPONENT core_tools)
