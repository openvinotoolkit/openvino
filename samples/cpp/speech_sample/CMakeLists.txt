# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

file (GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
file (GLOB HDR ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp)
               ${CMAKE_CURRENT_SOURCE_DIR}/*.h)

# Required zlib and cnpy dependencies

find_package(ZLIB QUIET)

if(NOT TARGET ZLIB::ZLIB)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_search_module(zlib QUIET
                          IMPORTED_TARGET GLOBAL
                          zlib)
        if(zlib_FOUND)
            add_library(ZLIB::ZLIB ALIAS PkgConfig::zlib)
        endif()
    endif()

    if(zlib_FOUND)
        message(STATUS "${PKG_CONFIG_EXECUTABLE}: zlib (${zlib_VERSION}) is found at ${zlib_PREFIX}")
    elseif(EXISTS "${Samples_SOURCE_DIR}/thirdparty/zlib")
        # OpenVINO package puts thirdparty to samples dir
        add_subdirectory("${Samples_SOURCE_DIR}/thirdparty/zlib"
                         "${Samples_BINARY_DIR}/thirdparty/zlib" EXCLUDE_FROM_ALL)
    elseif(EXISTS "${Samples_SOURCE_DIR}/../../thirdparty/zlib")
        # Allow running samples CMakeLists.txt as stand alone from openvino sources
        add_subdirectory("${Samples_SOURCE_DIR}/../../thirdparty/zlib"
                         "${Samples_BINARY_DIR}/thirdparty/zlib" EXCLUDE_FROM_ALL)
    endif()
endif()

if(EXISTS "${Samples_SOURCE_DIR}/thirdparty/cnpy")
    # OpenVINO package puts thirdparty to samples dir
    add_subdirectory("${Samples_SOURCE_DIR}/thirdparty/cnpy"
                     "${Samples_BINARY_DIR}/thirdparty/cnpy" EXCLUDE_FROM_ALL)
elseif(EXISTS "${Samples_SOURCE_DIR}/../../thirdparty/cnpy" AND NOT TARGET cnpy)
    # Allow running samples CMakeLists.txt as stand alone from openvino sources
    add_subdirectory("${Samples_SOURCE_DIR}/../../thirdparty/cnpy"
                     "${Samples_BINARY_DIR}/thirdparty/cnpy" EXCLUDE_FROM_ALL)
endif()

# add sample

ie_add_sample(NAME speech_sample
              SOURCES ${SRC}
              HEADERS ${HDR}
              DEPENDENCIES ${GFLAGS_TARGET} cnpy ie_samples_utils)
