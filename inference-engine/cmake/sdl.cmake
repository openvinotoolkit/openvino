# Copyright (C) 2018 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0
#

if (UNIX OR APPLE)
    set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} -fPIE -fPIC -Wformat -Wformat-security")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_FORTIFY_SOURCE=2")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -D_FORTIFY_SOURCE=2")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pie")
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -z noexecstack -z relro -z now")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -z noexecstack -z relro -z now")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
            set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} -fstack-protector-all")
        else()
            set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} -fstack-protector-strong")
        endif()
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(CMAKE_CCXX_FLAGS "${CMAKE_CCXX_FLAGS} -fstack-protector-all")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -z noexecstack -z relro -z now")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -z noexecstack -z relro -z now")
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_CCXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CCXX_FLAGS}")


elseif (WIN32)
    elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /sdl")

endif()

