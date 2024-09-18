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
    find_path(PNG_PKG_FILE
        NAMES libpng.pc
        HINTS  "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/png/lib/" "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/png/lib/pkgconfig/")
    if(PNG_PKG_FILE)
        file(STRINGS "${PNG_PKG_FILE}/libpng.pc"
            _png_lib_flags REGEX "^Libs.private:+.*")
        if(_png_lib_flags)
            string(REPLACE "Libs.private: "
                "" png_lib_flags "${_png_lib_flags}")
        endif()
        message(STATUS "found libpng.pc in ${PNG_PKG_FILE}/libpng.pc, PNG_LIB_FLAGS=${png_lib_flags}, _png_lib_flags=${_png_lib_flags}") 
    else()
        message(STATUS "can not find libpng.pc") 
        set(png_lib_flags "")
    endif()

    set(PNG_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/png/lib/libpng.a" "${png_lib_flags}")
    set(PNG_FOUND TRUE)
    message(STATUS "PNG is found, PNG_INCLUDE_DIR=${PNG_INCLUDE_DIR}, PNG_LIBRARIES=${PNG_LIBRARIES}") 
else()
    message(STATUS "PNG is not found, diable png supported") 
endif()