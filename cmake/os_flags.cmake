# Copyright (C) 2018-2019 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

macro(disable_deprecated_warnings)
    if(WIN32)
        if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
            set(ie_cxx_deprecated "/Qdiag-disable:1478")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
            set(ie_cxx_deprecated "/wd4996")
        endif()
    else()
        set(ie_cxx_deprecated "-Wno-deprecated-declarations")
    endif()

    if(NOT ie_cxx_deprecated)
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ie_cxx_deprecated}")
endmacro()

macro(ie_enable_lto)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
    set(CMAKE_AR  "gcc-ar")
    set(CMAKE_RANLIB "gcc-ranlib")
endmacro()

if (WIN32)
    set_property(DIRECTORY APPEND PROPERTY COMPILE_DEFINITIONS _CRT_SECURE_NO_WARNINGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS")
    # w2586 decorated name length exceeded, name was truncated

    if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd2586")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /bigobj /std:c++14")
        # Error        C4703 potentially uninitialized local pointer variable used
        # Error        C4996 declared deprecated
        # Error        C4146 unary minus operator applied to unsigned type, result still unsigned
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4146 /wd4996 /wd4703")

    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc") #no asynchronous structured exception handling
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")

    if (TREAT_WARNING_AS_ERROR)
        if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qdiag-warning:2586,177,3180,1740,1786,47,161")
        elseif (CMAKE_CXX_COMPILER_ID MATCHES MSVC)
#            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /WX") # Too many warnings
        endif()
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES Intel)
        # 161 unrecognized pragma
        # 177 variable was declared but never referenced
        # 2586 decorated name length exceeded, name was truncated
        # 3180 unrecognized OpenMP pragma
        # 11075: To get full report use -Qopt-report:4 -Qopt-report-phase ipo
        # 15335 was not vectorized: vectorization possible but seems inefficient. Use vector always directive or /Qvec-threshold0 to override
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qdiag-disable:161,177,2586,3180,11075,15335")
    endif()

    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} /Z7")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Z7")

    if(ENABLE_DEBUG_SYMBOLS)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Z7")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Z7")

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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Werror=return-type ")
    if (APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=unused-command-line-argument")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-private-field")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-reorder")       
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wswitch")    
    elseif(UNIX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized -Winit-self")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
        else()
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wmaybe-uninitialized")
        endif()
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -diag-disable=remark")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")

    if(LINUX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,--exclude-libs,ALL")
    endif()
endif()
