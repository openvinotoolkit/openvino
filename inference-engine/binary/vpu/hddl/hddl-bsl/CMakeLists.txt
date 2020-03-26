# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 2.6)

project(hddl-bsl)

if (NOT CMAKE_BUILD_TYPE)
    message("Build type not specified, setting to Release")
    set(CMAKE_BUILD_TYPE RELEASE)
endif ()

if (NOT MSVC)
    if (CMAKE_VERSION VERSION_LESS "3.1")
        if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
            set(CMAKE_C_FLAGS "-std=gnu99 ${CMAKE_C_FLAGS}")
        endif ()
    else ()
        set(CMAKE_C_STANDARD 99)
    endif ()

    set(SECURE_CFLAGS "-fPIE -fPIC -Wformat -Wformat-security")
    set(SECURE_CFLAGS "${SECURE_CFLAGS} -D_FORTIFY_SOURCE=2")
    if (CMAKE_C_COMPILER_VERSION VERSION_GREATER 4.8)
        set(SECURE_CFLAGS "${SECURE_CFLAGS} -fstack-protector-strong")
    else ()
        set(SECURE_CFLAGS "${SECURE_CFLAGS} -fstack-protector")
    endif ()
    set(SECURE_LDFLAGS "-ldl -z noexecstack -z relro -z now")
    set(CMAKE_C_FLAGS_DEBUG "-g3 -fsanitize=address -O0")
    set(CMAKE_C_FLAGS_RELEASE "-O2")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SECURE_CFLAGS} -Wall")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_C_LDFLAGS} ${SECURE_LDFLAGS} -pie")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_C_LDFLAGS} ${SECURE_LDFLAGS}")
else ()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /sdl")
    set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} /sdl")
endif ()

if (DEFINED CMAKE_INSTALL_PREFIX)
    message(STATUS "INSTALL_PATH is '${CMAKE_INSTALL_PREFIX}'.")
else ()
    if (NOT MSVC)
        set(CMAKE_INSTALL_PREFIX "/usr/local")
    endif ()
endif ()

include_directories("./include")

if (MSVC)
    add_subdirectory(json-c)
endif ()
add_subdirectory(src)

execute_process(
        COMMAND git log -1 --format=%h
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(TIMESTAMP DATE "%Y%m%d")
set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${DATE}-${GIT_COMMIT_HASH}-${CMAKE_SYSTEM_NAME}")
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
set(CPACK_ARCHIVE_COMPONENT_INSTALL TRUE)
set(CPACK_COMPONENTS_ALL usbrules_generator libraries headers)
set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)

if (UNIX)
    set(CPACK_GENERATOR "TGZ")
else ()
    set(CPACK_GENERATOR "ZIP")
endif ()

include(CPack)
