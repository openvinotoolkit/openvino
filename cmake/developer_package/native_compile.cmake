# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(ExternalProject)

#
# ov_native_compile_external_project(
#      TARGET_NAME <name>
#      NATIVE_SOURCE_DIR <source dir>
#      NATIVE_BUILD_DIR <source dir>
#      NATIVE_INSTALL_DIR <source dir>
#      NATIVE_PREFIX_DIR <source dir>
#      CMAKE_OPTION <option1 option2 ...>
#   )
#
function(ov_native_compile_external_project)
    set(oneValueRequiredArgs NATIVE_SOURCE_DIR NATIVE_BUILD_DIR
                             NATIVE_INSTALL_DIR NATIVE_PREFIX_DIR
                             TARGET_NAME)
    set(multiValueArgs CMAKE_OPTION)
    cmake_parse_arguments(ARG "" "${oneValueRequiredArgs};${oneValueOptionalArgs}" "${multiValueArgs}" ${ARGN})

    if(YOCTO_AARCH64)
        # need to unset several variables which can set env to cross-environment
        foreach(var SDKTARGETSYSROOT CONFIG_SITE OECORE_NATIVE_SYSROOT OECORE_TARGET_SYSROOT
                    OECORE_ACLOCAL_OPTS OECORE_BASELIB OECORE_TARGET_ARCH OECORE_TARGET_OS CC CXX
                    CPP AS LD GDB STRIP RANLIB OBJCOPY OBJDUMP READELF AR NM M4 TARGET_PREFIX
                    CONFIGURE_FLAGS CFLAGS CXXFLAGS LDFLAGS CPPFLAGS KCFLAGS OECORE_DISTRO_VERSION
                    OECORE_SDK_VERSION ARCH CROSS_COMPILE OE_CMAKE_TOOLCHAIN_FILE OPENSSL_CONF
                    OE_CMAKE_FIND_LIBRARY_CUSTOM_LIB_SUFFIX PKG_CONFIG_SYSROOT_DIR PKG_CONFIG_PATH)
            if(DEFINED ENV{${var}})
                list(APPEND cmake_env --unset=${var})
            endif()
        endforeach()

        # filter out PATH from yocto locations
        string(REPLACE ":" ";" custom_path "$ENV{PATH}")
        foreach(path IN LISTS custom_path)
            if(NOT path MATCHES "^$ENV{OECORE_NATIVE_SYSROOT}")
                list(APPEND clean_path "${path}")
            endif()
        endforeach()

        find_host_program(NATIVE_CMAKE_COMMAND
                          NAMES cmake
                          PATHS ${clean_path}
                          DOC "Host cmake"
                          REQUIRED
                          NO_DEFAULT_PATH)
    else()
        set(NATIVE_CMAKE_COMMAND "${CMAKE_COMMAND}")
    endif()

    # if env has CMAKE_TOOLCHAIN_FILE, we need to skip it
    if(DEFINED ENV{CMAKE_TOOLCHAIN_FILE})
        list(APPEND cmake_env --unset=CMAKE_TOOLCHAIN_FILE)
    endif()

    # compile flags
    if(CMAKE_COMPILER_IS_GNUCXX)
        set(compile_flags "-Wno-undef -Wno-error")
    endif()

    ExternalProject_Add(${ARG_TARGET_NAME}
        SOURCE_DIR "${ARG_NATIVE_SOURCE_DIR}"
        CONFIGURE_COMMAND
            "${CMAKE_COMMAND}" -E env ${cmake_env}
                "${NATIVE_CMAKE_COMMAND}"
                # explicitly skip compiler and generator
                # "-DCMAKE_GENERATOR=${CMAKE_GENERATOR}"
                "-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}"
                "-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}"
                "-DCMAKE_CXX_LINKER_LAUNCHER=${CMAKE_CXX_LINKER_LAUNCHER}"
                "-DCMAKE_C_LINKER_LAUNCHER=${CMAKE_C_LINKER_LAUNCHER}"
                "-DCMAKE_CXX_FLAGS=${compile_flags}"
                "-DCMAKE_C_FLAGS=${compile_flags}"
                "-DCMAKE_POLICY_DEFAULT_CMP0069=NEW"
                "-DCMAKE_INSTALL_PREFIX=${ARG_NATIVE_INSTALL_DIR}"
                "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                "-DTHREADS_PREFER_PTHREAD_FLAG=${THREADS_PREFER_PTHREAD_FLAG}"
                # project specific
                ${ARG_CMAKE_OPTIONS}
                # source and build directories
                "-S${ARG_NATIVE_SOURCE_DIR}"
                "-B${ARG_NATIVE_BUILD_DIR}"
        BINARY_DIR "${ARG_NATIVE_BUILD_DIR}"
        INSTALL_DIR "${ARG_NATIVE_INSTALL_DIR}"
        PREFIX "${ARG_NATIVE_PREFIX_DIR}"
        EXCLUDE_FROM_ALL ON)
endfunction()
