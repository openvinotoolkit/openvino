# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    if(UNIX)
        set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} -Wformat -Wformat-security -D_FORTIFY_SOURCE=2")
        if(NOT APPLE)
            set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -pie")
        endif()

        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -z noexecstack -z relro -z now")
            set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -z noexecstack -z relro -z now")
            if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
                set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} -fstack-protector-all")
            else()
                set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} -fstack-protector-strong")
            endif()
            set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} -s")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} -fstack-protector-all")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} -fstack-protector-strong -Wl,--strip-all")
            set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} -z noexecstack -z relro -z now")
            set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -z noexecstack -z relro -z now")
        endif()
    elseif(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
        set(IE_C_CXX_FLAGS "${IE_C_CXX_FLAGS} /sdl")
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${IE_C_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${IE_C_CXX_FLAGS}")
endif()
