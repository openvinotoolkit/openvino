# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(opencv_c_wrapper)

set(TARGET_NAME ${PROJECT_NAME})

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
                  ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)
file(GLOB HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)

# create library
add_library(${TARGET_NAME} STATIC ${HEADERS} ${SOURCES})

# Find OpenCV components if exist
find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
if(NOT OpenCV_FOUND)
    message(WARNING "OPENCV is disabled or not found, ${TARGET_NAME} is built without OPENCV support")
else()
    target_compile_definitions(${TARGET_NAME} PRIVATE USE_OPENCV)
endif()

target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCV_LIBRARIES})

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
                                          PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER c_samples)

if(COMMAND add_clang_format_target AND NOT IE_SAMPLE_EXCLUDE_CLANG_FORMAT)
    add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})
endif()
