# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

PROJECT(zlib)

if(NOT WIN32)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-all")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-all")
endif()


if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  /MP /wd4996  /W3")
endif()

set(TARGET_NAME "zlib")

set(lib_srcs 
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
    zlib/zutil.c
)

set(lib_hdrs 
    zlib/crc32.h
    zlib/deflate.h
    zlib/gzguts.h
    zlib/inffast.h
    zlib/inffixed.h
    zlib/inflate.h
    zlib/inftrees.h
    zlib/trees.h
    zlib/zutil.h
)

set(lib_ext_hdrs "zlib/zlib.h" "zlib/zconf.h")
add_library(${TARGET_NAME} STATIC ${lib_srcs} ${lib_hdrs} ${lib_ext_hdrs})


target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/zlib"
                                                 "${CMAKE_CURRENT_SOURCE_DIR}/zlib/..")

set_target_properties(zlib PROPERTIES FOLDER thirdparty)