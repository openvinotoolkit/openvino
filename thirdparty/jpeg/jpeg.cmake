# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
execute_process(COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/jpeg/build.sh"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/jpeg/"
    RESULT_VARIABLE result_var
    OUTPUT_VARIABLE output_var
    ERROR_VARIABLE error_var)
message(STATUS "result_var=${result_var}")
if(result_var STREQUAL  "0")
    set(JPEG_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/jpeg/include)
    set(JPEG_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/jpeg/lib/libjpeg.a)
    set(JPEG_FOUND TRUE)
    message(STATUS "JPEG is found, JPEG_INCLUDE_DIR=${JPEG_INCLUDE_DIR}, JPEG_LIBRARIES=${JPEG_LIBRARIES}") 
else()
    message(STATUS "PNG is not found, diable png supported") 
endif()
