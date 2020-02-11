# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(OpenCV_C_Wraper)
set(TARGET_NAME opencv_c_wraper)

#set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--exclude-libs,ALL")
file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

# create library
add_library(${TARGET_NAME} SHARED ${HEADERS} ${SOURCES})

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS imgcodecs videoio  QUIET)
if(NOT OpenCV_FOUND)
    message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " is built without OPENCV support")
else()
    add_definitions(-DUSE_OPENCV)
endif()

target_link_libraries(${TARGET_NAME} PUBLIC ${OpenCV_LIBRARIES})

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

if(COMMAND add_cpplint_target)
	add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})
endif()
