# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set (TARGET_NAME "timetests_helper")

file (GLOB SRC *.cpp)
add_library(${TARGET_NAME} STATIC ${SRC})
target_include_directories(${TARGET_NAME} PUBLIC "${CMAKE_SOURCE_DIR}/include")

find_package(gflags QUIET)
if (gflags_FOUND)
    set(GFLAGS_LIBRARIES gflags)  # use gflags from developer package
else()
    include(ExternalProject)
    find_package(Threads)
    set(gflags_PREFIX ${CMAKE_BINARY_DIR}/external/gflags-prefix)
    set(gflags_INSTALL ${CMAKE_BINARY_DIR}/external/gflags-install)
    set(gflags_LIB ${gflags_INSTALL}/lib/libgflags.a)

    ExternalProject_Add(
        gflags
        PREFIX ${gflags_PREFIX}
        GIT_REPOSITORY "https://github.com/gflags/gflags.git"
        GIT_TAG "v2.2.2"
        UPDATE_COMMAND ""
        INSTALL_DIR ${gflags_INSTALL}
        CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}
        CMAKE_GENERATOR ${CMAKE_GENERATOR}
        CMAKE_GENERATOR_PLATFORM ${CMAKE_GENERATOR_PLATFORM}
        CMAKE_GENERATOR_TOOLSET ${CMAKE_GENERATOR_TOOLSET}
        CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${gflags_INSTALL}
        EXCLUDE_FROM_ALL TRUE
        BUILD_BYPRODUCTS ${gflags_LIB}
        LOG_DOWNLOAD 1
        LOG_INSTALL 1
    )
    set(GFLAGS_LIBRARIES ${gflags_LIB} ${CMAKE_THREAD_LIBS_INIT})
    add_dependencies(${TARGET_NAME} gflags)
    target_include_directories(${TARGET_NAME} PRIVATE "${gflags_INSTALL}/include")
endif()

target_link_libraries(${TARGET_NAME} ${GFLAGS_LIBRARIES})
