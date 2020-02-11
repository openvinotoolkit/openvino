# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Disables deprecated warnings generation
# Defines ie_c_cxx_deprecated varaible which contains C / C++ compiler flags
#
macro(disable_deprecated_warnings)
    if(WIN32)
        if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
            set(ie_c_cxx_deprecated "/Qdiag-disable:1478,1786")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
            set(ie_c_cxx_deprecated "/wd4996")
        endif()
    else()
        set(ie_c_cxx_deprecated "-Wno-deprecated-declarations")
    endif()

    if(NOT ie_c_cxx_deprecated)
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ie_c_cxx_deprecated}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ie_c_cxx_deprecated}")
endmacro()

#
# Enables Link Time Optimization compilation
#
macro(ie_enable_lto)
    if(UNIX)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -flto")
        # LTO causes issues with gcc 4.8.5 during cmake pthread check
        if(NOT CMAKE_C_COMPILER_VERSION VERSION_LESS 4.9)
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -flto")
        endif()

        # modify linker and ar
        if(LINUX)
            set(CMAKE_AR  "gcc-ar")
            set(CMAKE_RANLIB "gcc-ranlib")
        endif()
    elseif(WIN32)
        if(CMAKE_BUILD_TYPE STREQUAL Release)
            # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /GL")
            # set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /GL")
            # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG:STATUS")
            # set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /LTCG:STATUS")
        endif()
    endif()
endmacro()

#
# Adds compiler flags to C / C++ sources
#
macro(ie_add_compiler_flags)
    foreach(flag ${ARGN})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}")
    endforeach()
endmacro()

#
# Compilation and linker flags
#

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsigned-char")
endif()

if(WIN32)
    ie_add_compiler_flags(-D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
    ie_add_compiler_flags(/EHsc) # no asynchronous structured exception handling
    ie_add_compiler_flags(/Gy) # remove unreferenced functions: function level linking
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")

    if (TREAT_WARNING_AS_ERROR)
        if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
            ie_add_compiler_flags(/WX)
            ie_add_compiler_flags(/Qdiag-warning:47,1740,1786)
        elseif (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
           # ie_add_compiler_flags(/WX) # Too many warnings
        endif()
    endif()

    # Compiler specific flags

    ie_add_compiler_flags(/bigobj)
    if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
        ie_add_compiler_flags(/MP /std:c++14)
    endif()

    # Disable noisy warnings

    if(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
        # C4251 needs to have dll-interface to be used by clients of class
        ie_add_compiler_flags(/wd4251)
        # C4275 non dll-interface class used as base for dll-interface class
        ie_add_compiler_flags(/wd4275)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
        # 161 unrecognized pragma
        # 177 variable was declared but never referenced
        # 2586 decorated name length exceeded, name was truncated
        # 2651: attribute does not apply to any entity
        # 3180 unrecognized OpenMP pragma
        # 11075: To get full report use -Qopt-report:4 -Qopt-report-phase ipo
        # 15335 was not vectorized: vectorization possible but seems inefficient. Use vector always directive or /Qvec-threshold0 to override
        ie_add_compiler_flags(/Qdiag-disable:161,177,2586,2651,3180,11075,15335)
    endif()

    # Debug information flags

    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /Z7")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Z7")

    if(ENABLE_DEBUG_SYMBOLS)
        ie_add_compiler_flags(/Z7)

        set(DEBUG_SYMBOLS_LINKER_FLAGS "/DEBUG")
        if (CMAKE_BUILD_TYPE STREQUAL "Release")
            # Keep default /OPT values. See /DEBUG reference for details.
            set(DEBUG_SYMBOLS_LINKER_FLAGS "${DEBUG_SYMBOLS_LINKER_FLAGS} /OPT:REF /OPT:ICF")
        endif()

        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${DEBUG_SYMBOLS_LINKER_FLAGS}")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${DEBUG_SYMBOLS_LINKER_FLAGS}")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${DEBUG_SYMBOLS_LINKER_FLAGS}")
    endif()
else()
    # TODO: enable for C sources as well
    # ie_add_compiler_flags(-Werror)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    ie_add_compiler_flags(-ffunction-sections -fdata-sections)
    ie_add_compiler_flags(-fvisibility=hidden)
    ie_add_compiler_flags(-fdiagnostics-show-option)
    ie_add_compiler_flags(-Wundef)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")

    # Disable noisy warnings

    if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        ie_add_compiler_flags(-Wswitch)
    elseif(UNIX)
        ie_add_compiler_flags(-Wuninitialized -Winit-self)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            ie_add_compiler_flags(-Wno-error=switch)
        else()
            ie_add_compiler_flags(-Wmaybe-uninitialized)
        endif()
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        ie_add_compiler_flags(-diag-disable=remark)
    endif()

    # Linker flags

    if(APPLE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-dead_strip")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-dead_strip")
    elseif(LINUX)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,--exclude-libs,ALL")
    endif()
endif()
