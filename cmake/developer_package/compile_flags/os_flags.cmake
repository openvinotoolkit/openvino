# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(compile_flags/functions)

#
# Compilation and linker flags
#

if(NOT DEFINED CMAKE_POSITION_INDEPENDENT_CODE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# to allows to override CMAKE_CXX_STANDARD from command line
if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
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

    # Specifies both the source character set and the execution character set as UTF-8.
    # For details, refer to link: https://learn.microsoft.com/en-us/cpp/build/reference/utf-8-set-source-and-executable-character-sets-to-utf-8?view=msvc-170
    ov_add_compiler_flags(/utf-8)

    # Workaround for an MSVC compiler issue in some versions of Visual Studio 2022.
    # The issue involves a null dereference to a mutex. For details, refer to link https://github.com/microsoft/STL/wiki/Changelog#vs-2022-1710
    if(MSVC AND MSVC_VERSION GREATER_EQUAL 1930)
        ov_add_compiler_flags(/D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
    endif()

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
    # PDB related flags
    #

    if(ENABLE_PDB_IN_RELEASE)
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Zi")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Zi")

        set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DEBUG")
        set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} /DEBUG")
        set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /DEBUG")
    endif()

    # need to set extra flags after /DEBUG to ensure that binary size is not bloated
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /OPT:REF /OPT:ICF")
    set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} /OPT:REF /OPT:ICF")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /OPT:REF /OPT:ICF")

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
