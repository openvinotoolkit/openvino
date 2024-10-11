# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(ProcessorCount)
include(CheckCXXCompilerFlag)

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
# ov_sse42_optimization_flags(<output flags>)
#
# Provides SSE4.2 compilation flags depending on an OS and a compiler
#
macro(ov_sse42_optimization_flags flags)
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
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" OR CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    elseif(ANDROID)
        if(ANDROID_ABI STREQUAL "arm64-v8a")
            set(${flags} -march=armv8.2-a+fp16 -Wno-unused-command-line-argument)
        else()
            message(WARNING "fp16 is not supported by Android armv7")
        endif()
    elseif(AARCH64)
        set(${flags} -O2 -march=armv8.2-a+fp16)
        if(NOT CMAKE_CL_64)
            list(APPEND ${flags} -ftree-vectorize)
        endif()
    elseif(ARM)
        message(WARNING "fp16 is not supported by 32-bit ARM")
    else()
        message(WARNING "fp16 is not supported by architecture ${CMAKE_SYSTEM_PROCESSOR}")
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
# Compilation and linker flags
#

if(NOT DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# to allows to override CMAKE_CXX_STANDARD from command line
if(NOT DEFINED CMAKE_CXX_STANDARD)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_STANDARD 14)
    elseif(OV_COMPILER_IS_INTEL_LLVM)
        set(CMAKE_CXX_STANDARD 17)
    else()
        set(CMAKE_CXX_STANDARD 11)
    endif()
endif()

if(NOT DEFINED CMAKE_CXX_EXTENSIONS)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

if(NOT DEFINED CMAKE_CXX_STANDARD_REQUIRED)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(ENABLE_COVERAGE)
    ov_add_compiler_flags(--coverage)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()

# Honor visibility properties for all target types
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(CMAKE_CL_64)
    # Default char Type Is unsigned
    # ov_add_compiler_flags(/J)
elseif(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG OR (OV_COMPILER_IS_INTEL_LLVM AND UNIX))
    ov_add_compiler_flags(-fsigned-char)
endif()

file(RELATIVE_PATH OV_RELATIVE_BIN_PATH ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_SOURCE_DIR})

if(CMAKE_VERSION VERSION_LESS 3.20)
    file(TO_NATIVE_PATH ${CMAKE_SOURCE_DIR} OV_NATIVE_PROJECT_ROOT_DIR)
    file(TO_NATIVE_PATH ${OV_RELATIVE_BIN_PATH} NATIVE_OV_RELATIVE_BIN_PATH)
else()
    cmake_path(NATIVE_PATH CMAKE_SOURCE_DIR OV_NATIVE_PROJECT_ROOT_DIR)
    cmake_path(NATIVE_PATH OV_RELATIVE_BIN_PATH NATIVE_OV_RELATIVE_BIN_PATH)
endif()

file(RELATIVE_PATH OV_NATIVE_PARENT_PROJECT_ROOT_DIR "${CMAKE_SOURCE_DIR}/.." ${CMAKE_SOURCE_DIR})

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    #
    # Common options / warnings enabled
    #

    ov_add_compiler_flags(/D_CRT_SECURE_NO_WARNINGS /D_SCL_SECURE_NO_WARNINGS)
     # no asynchronous structured exception handling
    ov_add_compiler_flags(/EHsc)
    # Allows the compiler to package individual functions in the form of packaged functions (COMDATs).
    ov_add_compiler_flags(/Gy)
    # This option helps ensure the fewest possible hard-to-find code defects. Similar to -Wall on GNU / Clang
    ov_add_compiler_flags(/W3)

    # Increase Number of Sections in .Obj file
    ov_add_compiler_flags(/bigobj)
    # Build with multiple processes
    ov_add_compiler_flags(/MP)

    if(AARCH64 AND NOT MSVC_VERSION LESS 1930)
        # otherwise, _ARM64_EXTENDED_INTRINSICS is defined, which defines 'mvn' macro
        ov_add_compiler_flags(/D_ARM64_DISTINCT_NEON_TYPES)
    endif()

    # Handle Large Addresses
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")

    #
    # Warnings as errors
    #

    if(CMAKE_COMPILE_WARNING_AS_ERROR)
        if(CMAKE_VERSION VERSION_LESS 3.24)
            ov_add_compiler_flags(/WX)
        endif()
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /WX")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /WX")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /WX")
    endif()

    #
    # Disable noisy warnings
    #

    # C4251 needs to have dll-interface to be used by clients of class
    ov_add_compiler_flags(/wd4251)
    # C4275 non dll-interface class used as base for dll-interface class
    ov_add_compiler_flags(/wd4275)

    # Enable __FILE__ trim, use path with forward and backward slash as directory separator
    # github actions use sccache which doesn't support /d1trimfile compile option
    if(NOT DEFINED ENV{GITHUB_ACTIONS})
        add_compile_options(
            "$<$<COMPILE_LANGUAGE:CXX>:/d1trimfile:${OV_NATIVE_PROJECT_ROOT_DIR}\\>"
            "$<$<COMPILE_LANGUAGE:CXX>:/d1trimfile:${CMAKE_SOURCE_DIR}/>")
    endif()

    #
    # Debug information flags, by default CMake adds /Zi option
    # but provides no way to specify CMAKE_COMPILE_PDB_NAME on root level
    # In order to avoid issues with ninja we are replacing default flag instead of having two of them
    # and observing warning D9025 about flag override
    #

    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
elseif(OV_COMPILER_IS_INTEL_LLVM AND WIN32)
    #
    # Warnings as errors
    #

    ov_add_compiler_flags(/WX)

    #
    # Disable noisy warnings
    #
    ov_disable_deprecated_warnings()
else()
    #
    # Common enabled warnings
    #

    # allow linker eliminating the unused code and data from the final executable
    ov_add_compiler_flags(-ffunction-sections -fdata-sections)
    # emits text showing the command-line option controlling a diagnostic
    ov_add_compiler_flags(-fdiagnostics-show-option)

    # This enables all the warnings about constructions that some users consider questionable, and that are easy to avoid
    ov_add_compiler_flags(-Wall)
    # Warn if an undefined identifier is evaluated in an #if directive. Such identifiers are replaced with zero.
    ov_add_compiler_flags(-Wundef)

    # To guarantee OpenVINO can be used with gcc versions 7 through 12
    # - https://gcc.gnu.org/onlinedocs/gcc/C_002b_002b-Dialect-Options.html
    # - https://gcc.gnu.org/onlinedocs/libstdc++/manual/abi.html
    if((CMAKE_COMPILER_IS_GNUCXX AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8) OR
       (OV_COMPILER_IS_CLANG AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 10) OR OV_COMPILER_IS_INTEL_LLVM)
        # Enable __FILE__ trim only for release mode
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ffile-prefix-map=${OV_NATIVE_PROJECT_ROOT_DIR}/= -ffile-prefix-map=${OV_RELATIVE_BIN_PATH}/=")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffile-prefix-map=${OV_NATIVE_PROJECT_ROOT_DIR}/= -ffile-prefix-map=${OV_RELATIVE_BIN_PATH}/=")
    endif()

    #
    # Warnings as errors
    #

    if(CMAKE_COMPILE_WARNING_AS_ERROR AND CMAKE_VERSION VERSION_LESS 3.24)
        ov_add_compiler_flags(-Werror)
    endif()

    #
    # Disable noisy warnings
    #

    if(OV_COMPILER_IS_INTEL_LLVM)
        ov_add_compiler_flags(-Wno-tautological-constant-compare)
        ov_disable_deprecated_warnings()
    endif()

    #
    # Linker flags
    #

    if(APPLE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-dead_strip")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-dead_strip")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-dead_strip")
    elseif(EMSCRIPTEN)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s MODULARIZE -s EXPORTED_RUNTIME_METHODS=ccall")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ERROR_ON_MISSING_LIBRARIES=1 -s ERROR_ON_UNDEFINED_SYMBOLS=1")
        # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=4")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s ALLOW_MEMORY_GROWTH=1")
        ov_add_compiler_flags(-sDISABLE_EXCEPTION_CATCHING=0)
        # ov_add_compiler_flags(-sUSE_PTHREADS=1)
    else()
        set(exclude_libs "-Wl,--exclude-libs,ALL")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections ${exclude_libs}")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--gc-sections ${exclude_libs}")
        if(NOT ENABLE_FUZZING)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${exclude_libs}")
        endif()
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
    endif()

    if(OV_COMPILER_IS_INTEL_LLVM)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static-intel")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-intel")
    endif()
endif()

add_compile_definitions(
    # Defines to trim check of __FILE__ macro in case if not done by compiler.
    OV_NATIVE_PARENT_PROJECT_ROOT_DIR="${OV_NATIVE_PARENT_PROJECT_ROOT_DIR}")

check_cxx_compiler_flag("-Wsuggest-override" SUGGEST_OVERRIDE_SUPPORTED)
if(SUGGEST_OVERRIDE_SUPPORTED)
    set(CMAKE_CXX_FLAGS "-Wsuggest-override ${CMAKE_CXX_FLAGS}")
endif()

check_cxx_compiler_flag("-Wunused-but-set-variable" UNUSED_BUT_SET_VARIABLE_SUPPORTED)

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
