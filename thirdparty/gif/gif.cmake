# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
execute_process(COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gif/build.sh"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gif/"
    RESULT_VARIABLE result_var
    OUTPUT_VARIABLE output_var
    ERROR_VARIABLE error_var)
message(STATUS "result_var=${result_var}")
if(result_var STREQUAL  "0")
    set(GIF_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gif/include)
    set(GIF_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gif/lib/libgif.a)
    set(GIF_FOUND TRUE)
    message(STATUS "GIF is found, GIF_INCLUDE_DIR=${GIF_INCLUDE_DIR}, GIF_LIBRARIES=${GIF_LIBRARIES}") 
else()
    message(STATUS "PNG is not found, diable png supported") 
endif()