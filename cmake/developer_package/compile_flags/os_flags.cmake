# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(ProcessorCount)
include(CheckCXXCompilerFlag)

#
# Disables deprecated warnings generation
# Defines ie_c_cxx_deprecated varaible which contains C / C++ compiler flags
#
macro(disable_deprecated_warnings)
    if(WIN32)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(ie_c_cxx_deprecated "/Qdiag-disable:1478,1786")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set(ie_c_cxx_deprecated "/wd4996")
        endif()
    else()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(ie_c_cxx_deprecated "-diag-disable=1478,1786")
        else()
            set(ie_c_cxx_deprecated "-Wno-deprecated-declarations")
        endif()
    endif()

    if(NOT ie_c_cxx_deprecated)
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${ie_c_cxx_deprecated}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${ie_c_cxx_deprecated}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ie_c_cxx_deprecated}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ie_c_cxx_deprecated}")
endmacro()

#
# Don't threat deprecated warnings as errors
# Defines ie_c_cxx_deprecated_no_errors varaible which contains C / C++ compiler flags
#
macro(ie_deprecated_no_errors)
    if(WIN32)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(ie_c_cxx_deprecated_no_errors "/Qdiag-warning:1478,1786")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            # show 4996 only for /w4
            set(ie_c_cxx_deprecated_no_errors "/wd4996")
            # WA for VPUX plugin
            set(ie_c_cxx_deprecated_no_errors "${ie_c_cxx_deprecated_no_errors} /wd4146 /wd4703")
        endif()
    else()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(ie_c_cxx_deprecated_no_errors "-diag-warning=1478,1786")
        else()
            set(ie_c_cxx_deprecated_no_errors "-Wno-error=deprecated-declarations")
        endif()

        if(NOT ie_c_cxx_deprecated_no_errors)
            message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
        endif()
    endif()

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${ie_c_cxx_deprecated_no_errors}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${ie_c_cxx_deprecated_no_errors}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ie_c_cxx_deprecated_no_errors}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ie_c_cxx_deprecated_no_errors}")
endmacro()

#
# Provides SSE4.2 compilation flags depending on an OS and a compiler
#
macro(ie_sse42_optimization_flags flags)
    if(WIN32)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            # No such option for MSVC 2019
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(${flags} /QxSSE4.2)
        else()
            message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
        endif()
    else()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(${flags} -xSSE4.2)
        else()
            set(${flags} -msse4.2)
        endif()
    endif()
endmacro()

#
# Provides AVX2 compilation flags depending on an OS and a compiler
#
macro(ie_avx2_optimization_flags flags)
    if(WIN32)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(${flags} /QxCORE-AVX2)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set(${flags} /arch:AVX2)
        else()
            message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
        endif()
    else()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(${flags} -xCORE-AVX2)
        else()
            set(${flags} -mavx2 -mfma)
        endif()
    endif()
endmacro()

#
# Provides common AVX512 compilation flags for AVX512F instruction set support
# depending on an OS and a compiler
#
macro(ie_avx512_optimization_flags flags)
    if(WIN32)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(${flags} /QxCOMMON-AVX512)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set(${flags} /arch:AVX512)
        else()
            message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
        endif()
    else()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(${flags} -xCOMMON-AVX512)
        endif()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set(${flags} -mavx512f -mfma)
        endif()
        if(CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang)$")
            set(${flags} -mavx512f -mfma)
        endif()
    endif()
endmacro()

macro(ie_arm_neon_optimization_flags flags)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # nothing
    elseif(ANDROID)
        if(ANDROID_ABI STREQUAL "arm64-v8a")
            set(${flags} -mfpu=neon)
        elseif(ANDROID_ABI STREQUAL "armeabi-v7a-hard with NEON")
            set(${flags} -march=armv7-a -mfloat-abi=hard -mhard-float -D_NDK_MATH_NO_SOFTFP=1 -mfpu=neon)
        elseif((ANDROID_ABI STREQUAL "armeabi-v7a with NEON") OR
               (ANDROID_ABI STREQUAL "armeabi-v7a" AND
                DEFINED CMAKE_ANDROID_ARM_NEON AND CMAKE_ANDROID_ARM_NEON))
            set(${flags} -march=armv7-a -mfloat-abi=softfp -mfpu=neon)
        endif()
    else()
        if(AARCH64)
            set(${flags} -O2 -ftree-vectorize)
        elseif(ARM)
            set(${flags} -mfpu=neon)
        endif()
    endif()
endmacro()

#
# Disables all warnings for 3rd party targets
#
function(ov_disable_all_warnings)
    foreach(target IN LISTS ARGN)
        get_target_property(target_type ${target} TYPE)

        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            target_compile_options(${target} PRIVATE /WX-)
        elseif(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
            target_compile_options(${target} PRIVATE -w)
            # required for LTO
            set(link_interface INTERFACE_LINK_OPTIONS)
            if(target_type STREQUAL "SHARED_LIBRARY" OR target_type STREQUAL "EXECUTABLE")
                set(link_interface LINK_OPTIONS)
            endif()
            set_target_properties(${target} PROPERTIES ${link_interface} "-Wno-error=maybe-uninitialized;-Wno-maybe-uninitialized")
        elseif(UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            # 193: zero used for undefined preprocessing identifier "XXX"
            # 1011: missing return statement at end of non-void function "XXX"
            # 2415: variable "xxx" of static storage duration was declared but never referenced
            target_compile_options(${target} PRIVATE -diag-disable=warn,193,1011,2415)
        endif()
    endforeach()
endfunction()

#
# Enables Link Time Optimization compilation
#
macro(ie_enable_lto)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
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

function(ov_add_compiler_flags)
    ie_add_compiler_flags(${ARGN})
endfunction()

#
# Forced includes certain header file to all target source files
#
function(ov_force_include target scope header_file)
    if(MSVC)
        target_compile_options(${target} ${scope} /FI"${header_file}")
    else()
        target_compile_options(${target} ${scope} -include "${header_file}")
    endif()
endfunction()

#
# Compilation and linker flags
#

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# to allows to override CMAKE_CXX_STANDARD from command line
if(NOT DEFINED CMAKE_CXX_STANDARD)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_STANDARD 14)
    else()
        set(CMAKE_CXX_STANDARD 11)
    endif()
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(ENABLE_COVERAGE)
    ie_add_compiler_flags(--coverage)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ie_add_compiler_flags(-fsigned-char)
endif()

# Honor visibility properties for all target types
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

function(ie_python_minimal_api target)
    # pybind11 uses a lot of API which is not a part of minimal python API subset
    # Ref 1: https://docs.python.org/3.11/c-api/stable.html
    # Ref 2: https://github.com/pybind/pybind11/issues/1755
    # target_compile_definitions(${target} PRIVATE Py_LIMITED_API=0x03090000)
    # if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    #     target_compile_options(${target} PRIVATE "-Wno-unused-variable")
    # endif()
endfunction()

if(WIN32)
    ie_add_compiler_flags(-D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
    ie_add_compiler_flags(/EHsc) # no asynchronous structured exception handling
    ie_add_compiler_flags(/Gy) # remove unreferenced functions: function level linking
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")

    if (CMAKE_COMPILE_WARNING_AS_ERROR)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            ie_add_compiler_flags(/Qdiag-warning:47,1740,1786)
        endif()
    endif()

    # Compiler specific flags

    ie_add_compiler_flags(/bigobj)
    ie_add_compiler_flags(/MP)

    # Disable noisy warnings

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # C4251 needs to have dll-interface to be used by clients of class
        ie_add_compiler_flags(/wd4251)
        # C4275 non dll-interface class used as base for dll-interface class
        ie_add_compiler_flags(/wd4275)
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        # 161: unrecognized pragma
        # 177: variable was declared but never referenced
        # 556: not matched type of assigned function pointer
        # 1744: field of class type without a DLL interface used in a class with a DLL interface
        # 1879: unimplemented pragma ignored
        # 2586: decorated name length exceeded, name was truncated
        # 2651: attribute does not apply to any entity
        # 3180: unrecognized OpenMP pragma
        # 11075: To get full report use -Qopt-report:4 -Qopt-report-phase ipo
        # 15335: was not vectorized: vectorization possible but seems inefficient. Use vector always directive or /Qvec-threshold0 to override
        ie_add_compiler_flags(/Qdiag-disable:161,177,556,1744,1879,2586,2651,3180,11075,15335)
    endif()

    # Debug information flags, by default CMake adds /Zi option
    # but provides no way to specify CMAKE_COMPILE_PDB_NAME on root level
    # In order to avoid issues with ninja we are replacing default flag instead of having two of them
    # and observing warning D9025 about flag override
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
else()
    ie_add_compiler_flags(-ffunction-sections -fdata-sections)
    ie_add_compiler_flags(-fdiagnostics-show-option)
    ie_add_compiler_flags(-Wundef)
    ie_add_compiler_flags(-Wreturn-type)
    ie_add_compiler_flags(-Wunused-variable)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        ie_add_compiler_flags(-Wswitch)
    elseif(UNIX)
        ie_add_compiler_flags(-Wuninitialized -Winit-self)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            ie_add_compiler_flags(-Winconsistent-missing-override
                                  -Wstring-plus-int)
        else()
            ie_add_compiler_flags(-Wmaybe-uninitialized)
            check_cxx_compiler_flag("-Wsuggest-override" SUGGEST_OVERRIDE_SUPPORTED)
            if(SUGGEST_OVERRIDE_SUPPORTED)
                set(CMAKE_CXX_FLAGS "-Wsuggest-override ${CMAKE_CXX_FLAGS}")
            endif()
        endif()
    endif()

    # Disable noisy warnings

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        # 177: function "XXX" was declared but never referenced
        ie_add_compiler_flags(-diag-disable=remark,177,2196)
    endif()

    # Linker flags

    if(APPLE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-dead_strip")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-dead_strip")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-dead_strip")
    elseif(LINUX)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,--exclude-libs,ALL")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--gc-sections -Wl,--exclude-libs,ALL")
        if(NOT ENABLE_FUZZING)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--exclude-libs,ALL")
        endif()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
    endif()
endif()

# Links provided libraries and include their INTERFACE_INCLUDE_DIRECTORIES as SYSTEM
function(link_system_libraries TARGET_NAME)
    set(MODE PRIVATE)

    foreach(arg IN LISTS ARGN)
        if(arg MATCHES "(PRIVATE|PUBLIC|INTERFACE)")
            set(MODE ${arg})
        else()
            if(TARGET "${arg}")
                target_include_directories(${TARGET_NAME}
                    SYSTEM ${MODE}
                        $<TARGET_PROPERTY:${arg},INTERFACE_INCLUDE_DIRECTORIES>
                        $<TARGET_PROPERTY:${arg},INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>
                )
            endif()

            target_link_libraries(${TARGET_NAME} ${MODE} ${arg})
        endif()
    endforeach()
endfunction()
