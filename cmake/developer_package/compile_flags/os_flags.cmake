# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(ProcessorCount)
include(CheckCXXCompilerFlag)

#
# ov_disable_deprecated_warnings()
#
# Disables deprecated warnings generation in current scope (directory, function)
# Defines ie_c_cxx_deprecated varaible which contains C / C++ compiler flags
#
macro(ov_disable_deprecated_warnings)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(ie_c_cxx_deprecated "/wd4996")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        if(WIN32)
            set(ie_c_cxx_deprecated "/Qdiag-disable:1478,1786")
        else()
            set(ie_c_cxx_deprecated "-diag-disable=1478,1786")
        endif()
    elseif(OV_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_GNUCXX)
        set(ie_c_cxx_deprecated "-Wno-deprecated-declarations")
    else()
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${ie_c_cxx_deprecated}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${ie_c_cxx_deprecated}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ie_c_cxx_deprecated}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ie_c_cxx_deprecated}")
endmacro()

macro(disable_deprecated_warnings)
    message(WARNING "disable_deprecated_warnings is deprecated, use ov_disable_deprecated_warnings instead")
    ov_disable_deprecated_warnings()
endmacro()

#
# ov_deprecated_no_errors()
#
# Don't threat deprecated warnings as errors in current scope (directory, function)
# Defines ie_c_cxx_deprecated_no_errors varaible which contains C / C++ compiler flags
#
macro(ov_deprecated_no_errors)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # show 4996 only for /w4
        set(ie_c_cxx_deprecated_no_errors "/wd4996")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        if(WIN32)
            set(ie_c_cxx_deprecated_no_errors "/Qdiag-warning:1478,1786")
        else()
            set(ie_c_cxx_deprecated_no_errors "-diag-warning=1478,1786")
        endif()
    elseif(OV_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_GNUCXX)
        set(ie_c_cxx_deprecated_no_errors "-Wno-error=deprecated-declarations")
        # Suppress #warning messages
        set(ie_c_cxx_deprecated_no_errors "${ie_c_cxx_deprecated_no_errors} -Wno-cpp")
    else()
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${ie_c_cxx_deprecated_no_errors}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${ie_c_cxx_deprecated_no_errors}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ie_c_cxx_deprecated_no_errors}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ie_c_cxx_deprecated_no_errors}")
endmacro()

#
# ov_dev_package_no_errors()
#
# Exports flags for 3rdparty modules, but without errors
#
macro(ov_dev_package_no_errors)
    if(OV_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_GNUCXX)
        set(ie_c_cxx_dev_no_errors "-Wno-all")
        if(SUGGEST_OVERRIDE_SUPPORTED)
            set(ie_cxx_dev_no_errors "-Wno-error=suggest-override")
        endif()
    endif()

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${ie_c_cxx_dev_no_errors} ${ie_cxx_dev_no_errors}")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${ie_c_cxx_dev_no_errors}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ie_c_cxx_dev_no_errors} ${ie_cxx_dev_no_errors}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ie_c_cxx_dev_no_errors}")
endmacro()

#
# ie_sse42_optimization_flags(<output flags>)
#
# Provides SSE4.2 compilation flags depending on an OS and a compiler
#
macro(ie_sse42_optimization_flags flags)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # No such option for MSVC 2019
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
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
# ie_avx2_optimization_flags(<output flags>)
#
# Provides AVX2 compilation flags depending on an OS and a compiler
#
macro(ie_avx2_optimization_flags flags)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(${flags} /arch:AVX2)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        if(WIN32)
            set(${flags} /QxCORE-AVX2)
        else()
            set(${flags} -xCORE-AVX2)
        endif()
    elseif(OV_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_GNUCXX)
        set(${flags} -mavx2 -mfma)
    else()
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()
endmacro()

#
# ie_avx512_optimization_flags(<output flags>)
#
# Provides common AVX512 compilation flags for AVX512F instruction set support
# depending on an OS and a compiler
#
macro(ie_avx512_optimization_flags flags)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(${flags} /arch:AVX512)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        if(WIN32)
            set(${flags} /QxCOMMON-AVX512)
        else()
            set(${flags} -xCOMMON-AVX512)
        endif()
    elseif(OV_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_GNUCXX)
        set(${flags} -mavx512f -mfma)
    else()
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()
endmacro()

#
# ie_arm_neon_optimization_flags(<output flags>)
#
macro(ie_arm_neon_optimization_flags flags)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
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
# ov_disable_all_warnings(<target1 [target2 target3 ...]>)
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
            if(CMAKE_COMPILER_IS_GNUCXX)
                set_target_properties(${target} PROPERTIES ${link_interface} "-Wno-error=maybe-uninitialized;-Wno-maybe-uninitialized")
            endif()
        elseif(UNIX AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            # 193: zero used for undefined preprocessing identifier "XXX"
            # 1011: missing return statement at end of non-void function "XXX"
            # 2415: variable "xxx" of static storage duration was declared but never referenced
            target_compile_options(${target} PRIVATE -diag-disable=warn,193,1011,2415)
        endif()
    endforeach()
endfunction()

#
# ie_enable_lto()
#
# Enables Link Time Optimization compilation
#
macro(ie_enable_lto)
    message(WARNING "ie_add_compiler_flags is deprecated, set INTERPROCEDURAL_OPTIMIZATION_RELEASE target property instead")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endmacro()

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

macro(ie_add_compiler_flags)
    message(WARNING "ie_add_compiler_flags is deprecated, use ov_add_compiler_flags instead")
    ov_add_compiler_flags(${ARGN})
endmacro()

#
# ov_force_include(<target> <PUBLIC | PRIVATE | INTERFACE> <header file>)
#
# Forced includes certain header file to all target source files
#
function(ov_force_include target scope header_file)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        target_compile_options(${target} ${scope} /FI"${header_file}")
    elseif(OV_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_GNUCXX)
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
    if(CMAKE_COMPILER_IS_GNUCXX AND NOT MINGW64)
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
elseif(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    ov_add_compiler_flags(-fsigned-char)
endif()

file(RELATIVE_PATH OV_RELATIVE_BIN_PATH ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_SOURCE_DIR})

if(${CMAKE_VERSION} VERSION_LESS "3.20")
    file(TO_NATIVE_PATH ${OpenVINO_SOURCE_DIR} OV_NATIVE_PROJECT_ROOT_DIR)
    file(TO_NATIVE_PATH ${OV_RELATIVE_BIN_PATH} NATIVE_OV_RELATIVE_BIN_PATH)
else()
    cmake_path(NATIVE_PATH OpenVINO_SOURCE_DIR OV_NATIVE_PROJECT_ROOT_DIR)
    cmake_path(NATIVE_PATH OV_RELATIVE_BIN_PATH NATIVE_OV_RELATIVE_BIN_PATH)
endif()

file(RELATIVE_PATH OV_NATIVE_PARENT_PROJECT_ROOT_DIR "${OpenVINO_SOURCE_DIR}/.." ${OpenVINO_SOURCE_DIR})

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
            "$<$<COMPILE_LANGUAGE:CXX>:/d1trimfile:${OpenVINO_SOURCE_DIR}/>")
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
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel" AND WIN32)
    #
    # Warnings as errors
    #

    if(CMAKE_COMPILE_WARNING_AS_ERROR AND CMAKE_VERSION VERSION_LESS 3.24)
        ov_add_compiler_flags(/Qdiag-warning:47,1740,1786)
    endif()

    #
    # Disable noisy warnings
    #

    # 161: unrecognized pragma
    ov_add_compiler_flags(/Qdiag-disable:161)
    # 177: variable was declared but never referenced
    ov_add_compiler_flags(/Qdiag-disable:177)
    # 556: not matched type of assigned function pointer
    ov_add_compiler_flags(/Qdiag-disable:556)
    # 1744: field of class type without a DLL interface used in a class with a DLL interface
    ov_add_compiler_flags(/Qdiag-disable:1744)
    # 1879: unimplemented pragma ignored
    ov_add_compiler_flags(/Qdiag-disable:1879)
    # 2586: decorated name length exceeded, name was truncated
    ov_add_compiler_flags(/Qdiag-disable:2586)
    # 2651: attribute does not apply to any entity
    ov_add_compiler_flags(/Qdiag-disable:2651)
    # 3180: unrecognized OpenMP pragma
    ov_add_compiler_flags(/Qdiag-disable:3180)
    # 11075: To get full report use -Qopt-report:4 -Qopt-report-phase ipo
    ov_add_compiler_flags(/Qdiag-disable:11075)
    # 15335: was not vectorized: vectorization possible but seems inefficient.
    # Use vector always directive or /Qvec-threshold0 to override
    ov_add_compiler_flags(/Qdiag-disable:15335)
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
    if(CMAKE_COMPILER_IS_GNUCXX)
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "8")
            # Enable __FILE__ trim only for release mode
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ffile-prefix-map=${OV_NATIVE_PROJECT_ROOT_DIR}/= -ffile-prefix-map=${OV_RELATIVE_BIN_PATH}/=")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffile-prefix-map=${OV_NATIVE_PROJECT_ROOT_DIR}/= -ffile-prefix-map=${OV_RELATIVE_BIN_PATH}/=")
        endif()
        # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wabi=11")
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL "10")
            # Enable __FILE__ trim only for release mode
            set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ffile-prefix-map=${OV_NATIVE_PROJECT_ROOT_DIR}/= -ffile-prefix-map=${OV_RELATIVE_BIN_PATH}/=")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffile-prefix-map=${OV_NATIVE_PROJECT_ROOT_DIR}/= -ffile-prefix-map=${OV_RELATIVE_BIN_PATH}/=")
        endif()
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

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        # 177: function "XXX" was declared but never referenced
        ov_add_compiler_flags(-diag-disable=remark,177,2196)
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
