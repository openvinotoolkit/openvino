# Copyright (C) 2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "statistics_collector")
set(TARGET_NAME_LIB "statistics_collector_s")

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS imgcodecs QUIET)
if(NOT(OpenCV_FOUND))
    message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " skipped")
    return()
endif()

file (GLOB SRC_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/data_stats.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/statistics_processor.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/image_decoder.cpp
        )
file (GLOB PUBLIC_HEADERS_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
        )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SRC_LIB})
source_group("include" FILES ${PUBLIC_HEADERS_LIB})

include_directories (
    ${IE_MAIN_SOURCE_DIR}/samples
    ${IE_MAIN_SOURCE_DIR}/samples/common
    ${IE_MAIN_SOURCE_DIR}/thirdparty
    ${IE_MAIN_SOURCE_DIR}/include
    ${IE_MAIN_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR}/samples/thirdparty/gflags/include
)

# Create library file from sources.
add_library(${TARGET_NAME_LIB} STATIC ${SRC_LIB} ${PUBLIC_HEADERS_LIB})
add_executable(${TARGET_NAME} main.cpp)

add_dependencies(${TARGET_NAME} gflags)

set_target_properties(${TARGET_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        COMPILE_PDB_NAME ${TARGET_NAME})

target_include_directories(${TARGET_NAME_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${TARGET_NAME_LIB} PUBLIC inference_engine ${OpenCV_LIBRARIES})
target_link_libraries(${TARGET_NAME} ${TARGET_NAME_LIB} gflags)

if(UNIX)
    target_link_libraries(${TARGET_NAME} ${LIB_DL} pthread)
endif()
set_ie_threading_interface_for(${TARGET_NAME_LIB})

# export for python

export(TARGETS ${TARGET_NAME_LIB} NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets_developer.cmake")
