# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

execute_process(COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/png/build.sh"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/png/"
    RESULT_VARIABLE result_var
    OUTPUT_VARIABLE output_var
    ERROR_VARIABLE error_var)
message(STATUS "result_var=${result_var}")
if(result_var STREQUAL  "0")
    set(PNG_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/png/include)
    set(PNG_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/png/lib/libpng.so)
    set(PNG_FOUND TRUE)
    message(STATUS "PNG is found, PNG_INCLUDE_DIR=${PNG_INCLUDE_DIR}, PNG_LIBRARIES=${PNG_LIBRARIES}") 
else()
    message(STATUS "PNG is not found, diable png supported") 
endif()