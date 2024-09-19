# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

function(ov_build_jpeg)
    set(BUILD_SHARED_LIBS OFF)
    set(WITH_JPEG8 TRUE)
    set(ENABLE_STATIC TRUE)
    set(ENABLE_SHARED FALSE)
    set(WITH_JAVA FALSE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # '<': signed/unsigned mismatch
        ov_add_compiler_flags(/wd4018)
        # conditional expression is constant
        ov_add_compiler_flags(/wd4127)
        # 'conversion' conversion from 'type1' to 'type2', possible loss of data
        ov_add_compiler_flags(/wd4244)
        # 'conversion' : conversion from 'type1' to 'type2', signed/unsigned mismatch
        ov_add_compiler_flags(/wd4245)
        # 'var' : conversion from 'size_t' to 'type', possible loss of data
        ov_add_compiler_flags(/wd4267)
    elseif(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR OV_COMPILER_IS_INTEL_LLVM)
        # we need to pass -Wextra first, then -Wno-sign-compare
        # otherwise, snappy's CMakeLists.txt will do it for us
        ov_add_compiler_flags(-Wextra)
        ov_add_compiler_flags(-Wno-sign-compare)
    endif()

    add_subdirectory(libjpeg EXCLUDE_FROM_ALL)
    add_library(jpeg::jpeg ALIAS jpeg-static)
endfunction()

include("${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/developer_package/download/extract.cmake")

extract(${CMAKE_CURRENT_SOURCE_DIR}/libjpeg.tgz ${CMAKE_CURRENT_SOURCE_DIR} TRUE "*" result_var)

if(result_var STREQUAL  "1")
    ov_build_jpeg()
    ov_install_static_lib(jpeg ${OV_CPACK_COMP_CORE})
    set(JPEG_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libjpeg" "${CMAKE_CURRENT_BINARY_DIR}/libjpeg" PARENT_SCOPE)
    set(JPEG_FOUND TRUE PARENT_SCOPE)
else()
    message(STATUS "Failed to build JPEG from thirdparty, disable JPEG supported. result_var=${result_var}") 
endif()