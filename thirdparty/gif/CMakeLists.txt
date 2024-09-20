# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
cmake_minimum_required(VERSION 3.1)
cmake_policy(VERSION 3.1)

project(gif C)

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -fPIC -Wall -Wno-format-truncation")

if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4996 /wd4244 /W3")
    set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /wd4995 /wd4244 /wd4996")
else()
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-all")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        #266: function "XXX" declared implicitly
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -diag-disable=266")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
        include(CheckCCompilerFlag)
        check_c_compiler_flag("-Wdeprecated-non-prototype" DEPRECATED_NO_PROTOTYPE)
        if(DEPRECATED_NO_PROTOTYPE)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-non-prototype")
        endif()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-unused-variable")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-deprecated-declarations")
    endif()
endif()

set(TARGET_NAME "gif-static")

set(gif_srcs
    giflib-5.1.9/dgif_lib.c
    giflib-5.1.9/egif_lib.c
    giflib-5.1.9/gifalloc.c
    giflib-5.1.9/gif_err.c
    giflib-5.1.9/gif_font.c
    giflib-5.1.9/gif_hash.c
    giflib-5.1.9/openbsd-reallocarray.c)


set(gif_hdrs
    giflib-5.1.9/gif_hash.h 
    giflib-5.1.9/gif_lib.h 
    giflib-5.1.9/gif_lib_private.h)


extract(${CMAKE_CURRENT_SOURCE_DIR}/giflib-5.1.9.tar.gz ${CMAKE_CURRENT_SOURCE_DIR} TRUE "*" result_var)

if(result_var STREQUAL  "1")
    add_library(${TARGET_NAME} STATIC ${gif_srcs} ${gif_hdrs})
    set_property(TARGET ${TARGET_NAME} PROPERTY COMPILE_FLAGS "-fPIC")
    set_target_properties(${TARGET_NAME} PROPERTIES OUTPUT_NAME gif)

    add_library(openvino::gif ALIAS ${TARGET_NAME})
    ov_developer_package_export_targets(TARGET openvino::gif)

    # ov_install_static_lib(${TARGET_NAME} ${OV_CPACK_COMP_CORE})

    foreach(comp IN ITEMS ${gif_hdrs})
        message(STATUS "COPY file ${comp} to ${CMAKE_CURRENT_BINARY_DIR}/include")
        file(COPY ${comp} DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/include")
    endforeach()

    set_target_properties(${TARGET_NAME} PROPERTIES FOLDER thirdparty)
    set(GIF_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include" PARENT_SCOPE)
    set(GIF_FOUND TRUE PARENT_SCOPE)
else()
    message(STATUS "Failed to build GIF from thirdparty, disable GIF supported. result_var=${result_var}") 
endif()



