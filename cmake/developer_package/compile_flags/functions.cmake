# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(ProcessorCount)
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

#
# ov_disable_deprecated_warnings()
#
# Disables deprecated warnings generation in current scope (directory, function)
# Defines ov_c_cxx_deprecated varaible which contains C / C++ compiler flags
#
macro(ov_disable_deprecated_warnings)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(ov_c_cxx_deprecated "/wd4996")
    elseif(OV_COMPILER_IS_INTEL_LLVM)
        if(WIN32)
            set(ov_c_cxx_deprecated "/Wno-deprecated-declarations")
        else()
            set(ov_c_cxx_deprecated "-Wno-deprecated-declarations")
        endif()
    elseif(OV_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_GNUCXX)
        set(ov_c_cxx_deprecated "-Wno-deprecated-declarations")
    else()
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${ov_c_cxx_deprecated}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${ov_c_cxx_deprecated}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ov_c_cxx_deprecated}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ov_c_cxx_deprecated}")
endmacro()

#
# ov_deprecated_no_errors()
#
# Don't threat deprecated warnings as errors in current scope (directory, function)
# Defines ov_c_cxx_deprecated_no_errors varaible which contains C / C++ compiler flags
#
macro(ov_deprecated_no_errors)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # show 4996 only for /w4
        set(ov_c_cxx_deprecated_no_errors "/wd4996")
    elseif(OV_COMPILER_IS_INTEL_LLVM)
        if(WIN32)
            set(ov_c_cxx_deprecated_no_errors "/Wno-error=deprecated-declarations")
        else()
            set(ov_c_cxx_deprecated_no_errors "-Wno-error=deprecated-declarations")
        endif()
    elseif(OV_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_GNUCXX)
        set(ov_c_cxx_deprecated_no_errors "-Wno-error=deprecated-declarations")
        # Suppress #warning messages
        set(ov_c_cxx_deprecated_no_errors "${ov_c_cxx_deprecated_no_errors} -Wno-cpp")
    else()
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${ov_c_cxx_deprecated_no_errors}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${ov_c_cxx_deprecated_no_errors}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ov_c_cxx_deprecated_no_errors}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ov_c_cxx_deprecated_no_errors}")
endmacro()

#
# ov_dev_package_no_errors()
#
# Exports flags for 3rdparty modules, but without errors
#
macro(ov_dev_package_no_errors)
    if(OV_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_GNUCXX OR (OV_COMPILER_IS_INTEL_LLVM AND UNIX))
        set(ov_c_cxx_dev_no_errors "-Wno-all")
        if(SUGGEST_OVERRIDE_SUPPORTED)
            set(ov_cxx_dev_no_errors "-Wno-error=suggest-override")
        endif()
    endif()

    if(CMAKE_COMPILE_WARNING_AS_ERROR AND WIN32)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR OV_COMPILER_IS_INTEL_LLVM)
            if(CMAKE_VERSION VERSION_LESS 3.24)
                ov_add_compiler_flags(/WX-)
            endif()
            string(REPLACE "/WX" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
        endif()
        set(CMAKE_COMPILE_WARNING_AS_ERROR OFF)
    endif()

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${ov_c_cxx_dev_no_errors} ${ov_cxx_dev_no_errors}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${ov_c_cxx_dev_no_errors}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ov_c_cxx_dev_no_errors} ${ov_cxx_dev_no_errors}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ov_c_cxx_dev_no_errors}")
endmacro()

#
# ov_check_compiler_supports_sve(flags)
#
# Checks whether CXX compiler for passed language supports SVE code compilation
#
macro(ov_check_compiler_supports_sve flags)
    # Code to compile
    set(SVE_CODE "
    #include <arm_sve.h>
    int main() {
        svfloat64_t a;
        a = svdup_n_f64(0);
        (void)a; // to avoid warnings
        return 0;
    }")

    # Save the current state of required flags
    set(CMAKE_REQUIRED_FLAGS_SAVE ${CMAKE_REQUIRED_FLAGS})

    # Set the flags necessary for compiling the test code with SVE support
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_CXX_FLAGS_INIT} ${flags}")

    # Check if the source code compiles with the given flags for C++
    CHECK_CXX_SOURCE_COMPILES("${SVE_CODE}" CXX_HAS_SVE)

    # If the compilation test is successful, set appropriate variables indicating support
    if(CXX_HAS_SVE)
        set(CXX_SVE_FOUND ON CACHE BOOL "SVE available on host")
        set(CXX_SVE_FOUND ON CACHE BOOL "CXX SVE support")
        set(CXX_SVE_FLAGS "${flags}" CACHE STRING "CXX SVE flags")
    endif()

    # Restore the original state of required flags
    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVE})

    # If the compilation test fails, indicate that the support is not found
    if(NOT CXX_SVE_FOUND)
        set(CXX_SVE_FOUND OFF CACHE BOOL "CXX SVE support")
        set(CXX_SVE_FLAGS "" CACHE STRING "CXX SVE flags")
    endif()

    # Mark the variables as advanced to hide them in the default CMake GUI
    mark_as_advanced(CXX_SVE_FOUND CXX_SVE_FLAGS)
endmacro()

#
# ov_sse42_optimization_flags(<output flags>)
#
# Provides SSE4.2 compilation flags depending on an OS and a compiler
#
macro(ov_sse42_optimization_flags flags)
    if(NOT ENABLE_SSE42)
        message(FATAL_ERROR "Internal error: ENABLE_SSE42 if OFF and 'ov_sse42_optimization_flags' must not be called")
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # No such option for MSVC 2019
    elseif(OV_COMPILER_IS_INTEL_LLVM)
        if(WIN32)
            set(${flags} /QxSSE4.2)
        else()
            set(${flags} -xSSE4.2)
        endif()
    elseif(OV_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_GNUCXX)
        set(${flags} -msse4.2)
        if(EMSCRIPTEN)
            list(APPEND ${flags} -msimd128)
        endif()
    else()
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()
endmacro()

#
# ov_avx2_optimization_flags(<output flags>)
#
# Provides AVX2 compilation flags depending on an OS and a compiler
#
macro(ov_avx2_optimization_flags flags)
    if(NOT ENABLE_AVX2)
        message(FATAL_ERROR "Internal error: ENABLE_AVX2 if OFF and 'ov_avx2_optimization_flags' must not be called")
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(${flags} /arch:AVX2)
    elseif(OV_COMPILER_IS_INTEL_LLVM)
        if(WIN32)
            set(${flags} /QxCORE-AVX2)
        else()
            set(${flags} -xCORE-AVX2)
        endif()
    elseif(OV_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_GNUCXX)
        set(${flags} -mavx2 -mfma -mf16c)
    else()
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()
endmacro()

#
# ov_avx512_optimization_flags(<output flags>)
#
# Provides common AVX512 compilation flags for AVX512F instruction set support
# depending on an OS and a compiler
#
macro(ov_avx512_optimization_flags flags)
    if(NOT ENABLE_AVX512F)
        message(FATAL_ERROR "Internal error: ENABLE_AVX512F if OFF and 'ov_avx512_optimization_flags' must not be called")
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(${flags} /arch:AVX512)
    elseif(OV_COMPILER_IS_INTEL_LLVM AND WIN32)
        set(${flags} /QxCORE-AVX512)
    elseif(OV_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_GNUCXX OR (OV_COMPILER_IS_INTEL_LLVM AND UNIX))
        set(${flags} -mavx512f -mavx512bw -mavx512vl -mfma -mf16c)
    else()
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()
endmacro()

#
# ov_arm_neon_optimization_flags(<output flags>)
#
macro(ov_arm_neon_optimization_flags flags)
    if(NOT (AARCH64 OR ARM))
        message(FATAL_ERROR "Internal error: platform is not ARM or AARCH64 and 'ov_arm_neon_optimization_flags' must not be called")
    endif()

    if(OV_COMPILER_IS_INTEL_LLVM)
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # nothing to define; works out of box
    elseif(ANDROID)
        if(ANDROID_ABI STREQUAL "arm64-v8a")
            set(${flags} -mfpu=neon -Wno-unused-command-line-argument)
        elseif(ANDROID_ABI STREQUAL "armeabi-v7a-hard with NEON")
            set(${flags} -march=armv7-a+fp -mfloat-abi=hard -mhard-float -D_NDK_MATH_NO_SOFTFP=1 -mfpu=neon -Wno-unused-command-line-argument)
        elseif((ANDROID_ABI STREQUAL "armeabi-v7a with NEON") OR
               (ANDROID_ABI STREQUAL "armeabi-v7a" AND
                DEFINED CMAKE_ANDROID_ARM_NEON AND CMAKE_ANDROID_ARM_NEON))
                set(${flags} -march=armv7-a+fp -mfloat-abi=softfp -mfpu=neon -Wno-unused-command-line-argument)
        endif()
    else()
        if(AARCH64)
            set(${flags} -O2)
            if(NOT CMAKE_CL_64)
                list(APPEND ${flags} -ftree-vectorize)
            endif()
        elseif(ARM)
            set(${flags} -mfpu=neon -Wno-unused-command-line-argument)
        endif()
    endif()
endmacro()

#
# ov_arm_neon_fp16_optimization_flags(<output flags>)
#
macro(ov_arm_neon_fp16_optimization_flags flags)
    if(NOT ENABLE_NEON_FP16)
        message(FATAL_ERROR "Internal error: ENABLE_NEON_FP16 if OFF and 'ov_arm_neon_fp16_optimization_flags' must not be called")
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID} for arm64 platform")
    elseif(ANDROID)
        if(ANDROID_ABI STREQUAL "arm64-v8a")
            set(${flags} -march=armv8.2-a+fp16 -Wno-unused-command-line-argument)
        else()
            message(WARNING "ARM64 fp16 is not supported by Android armv7")
        endif()
    elseif(AARCH64)
        set(${flags} -O2 -march=armv8.2-a+fp16)
        if(NOT CMAKE_CL_64)
            list(APPEND ${flags} -ftree-vectorize)
        endif()
    elseif(ARM)
        message(WARNING "ARM64 fp16 is not supported by 32-bit ARM")
    else()
        message(WARNING "ARM64 fp16 is not supported by architecture ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
endmacro()

#
# ov_arm_sve_optimization_flags(<output flags>)
#
macro(ov_arm_sve_optimization_flags flags)
    if(NOT ENABLE_SVE)
        message(FATAL_ERROR "Internal error: ENABLE_SVE if OFF and 'ov_arm_sve_optimization_flags' must not be called")
    endif()

    # Check for compiler SVE support
    ov_check_compiler_supports_sve("-march=armv8-a+sve")
    if(OV_COMPILER_IS_INTEL_LLVM)
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # nothing should be required here
    elseif(ANDROID)
        if(ANDROID_ABI STREQUAL "arm64-v8a")
            set(${flags} -Wno-unused-command-line-argument)
            if(CXX_SVE_FOUND)
                list(APPEND ${flags} -march=armv8-a+sve)
            else()
                message(WARNING "SVE is not supported on this Android ABI: ${ANDROID_ABI}")
            endif()
        else()
            message(WARNING "SVE is not supported on this Android ABI: ${ANDROID_ABI}")
        endif()
    else()
        if(AARCH64)
            set(${flags} -O2)

            # Add flag for SVE if supported
            if(CXX_SVE_FOUND)
                list(APPEND ${flags} -march=armv8-a+sve)
            endif()
            if(NOT CMAKE_CL_64)
                list(APPEND ${flags} -ftree-vectorize)
            endif()

            set(${flags} ${${flags}})
        elseif(ARM)
            message(WARNING "SVE is not supported on 32-bit ARM architectures.")
        else()
            message(WARNING "SVE is not supported by architecture ${CMAKE_SYSTEM_PROCESSOR}")
        endif()
    endif()
endmacro()

#
# ov_disable_all_warnings(<target1 [target2 target3 ...]>)
#
# Disables all warnings for 3rd party targets
#
function(ov_disable_all_warnings)
    foreach(target IN LISTS ARGN)
        get_target_property(target_type ${target} TYPE)

        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR (OV_COMPILER_IS_INTEL_LLVM AND WIN32))
            target_compile_options(${target} PRIVATE /WX-)
        elseif(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR (OV_COMPILER_IS_INTEL_LLVM AND UNIX))
            target_compile_options(${target} PRIVATE -w)
            # required for LTO
            set(link_interface INTERFACE_LINK_OPTIONS)
            if(target_type STREQUAL "SHARED_LIBRARY" OR target_type STREQUAL "EXECUTABLE")
                set(link_interface LINK_OPTIONS)
            endif()
            if(CMAKE_COMPILER_IS_GNUCXX)
                set_target_properties(${target} PROPERTIES ${link_interface} "-Wno-error=maybe-uninitialized;-Wno-maybe-uninitialized")
            endif()
        endif()
    endforeach()
endfunction()

#
# ov_add_compiler_flags(<flag1 [flag2 flag3 ...>])
#
# Adds compiler flags to C / C++ sources
#
macro(ov_add_compiler_flags)
    foreach(flag ${ARGN})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}")
    endforeach()
endmacro()

#
# ov_force_include(<target> <PUBLIC | PRIVATE | INTERFACE> <header file>)
#
# Forced includes certain header file to all target source files
#
function(ov_force_include target scope header_file)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR (OV_COMPILER_IS_INTEL_LLVM AND WIN32))
        target_compile_options(${target} ${scope} /FI"${header_file}")
    elseif(OV_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_GNUCXX OR (OV_COMPILER_IS_INTEL_LLVM AND UNIX))
        target_compile_options(${target} ${scope} -include "${header_file}")
    else()
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()
endfunction()

#
# ov_abi_free_target(<target name>)
#
# Marks target to be compiliance in CXX ABI free manner
#
function(ov_abi_free_target target)
    # To guarantee OpenVINO can be used with gcc versions 7 through 12.2
    # - https://gcc.gnu.org/onlinedocs/gcc/C_002b_002b-Dialect-Options.html
    # - https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html
    # Since gcc 13.0 abi=18
    # Version 18, which first appeard in G++ 13, fixes manglings of lambdas that have additional context.
    # which means we became not compatible
    if(CMAKE_COMPILER_IS_GNUCXX AND NOT MINGW64 AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13.0)
        target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wabi=11>)
    endif()
endfunction()

#
# ov_python_minimal_api(<target>)
#
# Set options to use only Python Limited API
#
function(ov_python_minimal_api target)
    # pybind11 uses a lot of API which is not a part of minimal python API subset
    # Ref 1: https://docs.python.org/3.11/c-api/stable.html
    # Ref 2: https://github.com/pybind/pybind11/issues/1755
    # target_compile_definitions(${target} PRIVATE Py_LIMITED_API=0x03090000)
    # if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    #     target_compile_options(${target} PRIVATE "-Wno-unused-variable")
    # endif()
endfunction()

#
# ov_link_system_libraries(target <PUBLIC | PRIVATE | INTERFACE> <lib1 [lib2 lib3 ...]>)
#
# Links provided libraries and include their INTERFACE_INCLUDE_DIRECTORIES as SYSTEM
#
function(ov_link_system_libraries TARGET_NAME)
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

#
# ov_try_use_gold_linker()
#
# Tries to use gold linker in current scope (directory, function)
#
function(ov_try_use_gold_linker)
    # don't use the gold linker, if the mold linker is set
    if(CMAKE_EXE_LINKER_FLAGS MATCHES "mold" OR CMAKE_MODULE_LINKER_FLAGS MATCHES "mold" OR CMAKE_SHARED_LINKER_FLAGS MATCHES "mold")
        return()
    endif()

    # gold linker on ubuntu20.04 may fail to link binaries build with sanitizer
    if(CMAKE_COMPILER_IS_GNUCXX AND NOT ENABLE_SANITIZER AND NOT CMAKE_CROSSCOMPILING)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fuse-ld=gold" PARENT_SCOPE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fuse-ld=gold" PARENT_SCOPE)
    endif()
endfunction()

#
# ov_target_link_libraries_as_system(<TARGET NAME> <PUBLIC | PRIVATE | INTERFACE> <target1 target2 ...>)
#
function(ov_target_link_libraries_as_system TARGET_NAME LINK_TYPE)
    target_link_libraries(${TARGET_NAME} ${LINK_TYPE} ${ARGN})

    # include directories as SYSTEM
    foreach(library IN LISTS ARGN)
        if(TARGET ${library})
            get_target_property(include_directories ${library} INTERFACE_INCLUDE_DIRECTORIES)
            if(include_directories)
                foreach(include_directory IN LISTS include_directories)
                    # cannot include /usr/include headers as SYSTEM
                    if(NOT "${include_directory}" MATCHES ".*/usr/include.*$")
                        # Note, some include dirs can be wrapper with $<BUILD_INTERFACE:dir1 dir2 ...> and we need to clean it
                        string(REGEX REPLACE "^\\$<BUILD_INTERFACE:" "" include_directory "${include_directory}")
                        string(REGEX REPLACE ">$" "" include_directory "${include_directory}")
                        target_include_directories(${TARGET_NAME} SYSTEM ${LINK_TYPE} $<BUILD_INTERFACE:${include_directory}>)
                    else()
                        set(_system_library ON)
                    endif()
                endforeach()
            endif()
        endif()
    endforeach()

    if(_system_library)
        # if we deal with system library (e.i. having /usr/include as header paths)
        # we cannot use SYSTEM key word for such library
        set_target_properties(${TARGET_NAME} PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
    endif()
endfunction()
