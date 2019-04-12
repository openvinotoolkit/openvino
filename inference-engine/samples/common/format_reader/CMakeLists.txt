# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "format_reader")

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

file (GLOB LIBRARY_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        )

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS imgcodecs videoio imgproc  QUIET)
if(NOT(OpenCV_FOUND))
    message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " is built without OPENCV support")
else()
    add_definitions(-DUSE_OPENCV)
endif()

add_definitions(-DIMPLEMENT_FORMAT_READER)

if(UNIX)
    list(REMOVE_ITEM MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/dllmain.cpp)
endif()
add_definitions(-DIMPLEMENT_INFERENCE_ENGINE_API)

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${LIBRARY_SRC})
source_group("include" FILES ${LIBRARY_HEADERS})


# Create library file from sources.
add_library(${TARGET_NAME} SHARED ${MAIN_SRC} ${LIBRARY_HEADERS})
target_link_libraries(${TARGET_NAME} ${OpenCV_LIBRARIES})

if(CMAKE_VERSION VERSION_LESS "2.8.11")
	include_directories (${CMAKE_CURRENT_SOURCE_DIR})
else()
	target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
endif()

set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_PDB_NAME ${TARGET_NAME})
