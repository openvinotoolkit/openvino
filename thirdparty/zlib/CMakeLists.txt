# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

project(zlib)

if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /wd4996 /wd4244 /W3")
    set (CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /wd4995 /wd4244 /wd4996")
else()
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-all")
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

set(TARGET_NAME "openvino_zlib")

set(zlib_srcs
    zlib/adler32.c
    zlib/compress.c
    zlib/crc32.c
    zlib/deflate.c
    zlib/gzclose.c
    zlib/gzlib.c
    zlib/gzread.c
    zlib/gzwrite.c
    zlib/inflate.c
    zlib/infback.c
    zlib/inftrees.c
    zlib/inffast.c
    zlib/trees.c
    zlib/uncompr.c
    zlib/zutil.c)

set(zlib_hdrs
    zlib/crc32.h
    zlib/deflate.h
    zlib/gzguts.h
    zlib/inffast.h
    zlib/inffixed.h
    zlib/inflate.h
    zlib/inftrees.h
    zlib/trees.h
    zlib/zutil.h)

set(zlib_ext_hdrs
    zlib/zlib.h
    zlib/zconf.h)

add_library(${TARGET_NAME} STATIC ${zlib_srcs} ${zlib_hdrs} ${lib_ext_hdrs})
add_library(openvino::zlib ALIAS ${TARGET_NAME})

target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/zlib")

set_target_properties(${TARGET_NAME} PROPERTIES FOLDER thirdparty)
