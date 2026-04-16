# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# This script resolves the prebuilt NPU Plugin Compiler dependency by downloading and extracting the appropriate
# archive based on the current platform. The expected location of the archive and naming convention is as follows:
#     vcl version: 7.6.0
#     release: releases/unified/2026/12_cip
#     storage location: https://storage.openvinotoolkit.org/dependencies/thirdparty
#     WINDOWS: 
#         windows2022: npu_compiler_vcl_windows_2022-7_6_0-da3cc32.zip
#     LINUX:
#         ubuntu22.04: npu_compiler_vcl_ubuntu_22_04-7_6_0-da3cc32.tar.gz
#         ubuntu24.04: npu_compiler_vcl_ubuntu_24_04-7_6_0-da3cc32.tar.gz
#
# This script replicates cmake/dependencies.cmake common OV dependency resolution logic including:
#     THIRDPARTY_SERVER_PATH environment variable or cmake options support that allows
#         to override default download location.
#     NPU_PLUGIN_COMPILER_ROOT environment variable support that allows to override default download and extraction
#         logic and point to already existing extracted archive location.
#     SHA256 checksum verification of the downloaded archive.
#     Checking the presence of the archive before downloading to avoid unnecessary downloads.
#
# To update the prebuilt compiler version, please update the following variables in this script:
#     PLUGIN_COMPILER_VERSION_MAJOR, PLUGIN_COMPILER_VERSION_MINOR,
#     PLUGIN_COMPILER_VERSION_PATCH, PLUGIN_COMPILER_COMMIT_SHA
#     PLUGIN_COMPILER_WINDOWS_2022_CHECKSUM, PLUGIN_COMPILER_UBUNTU_22_04_CHECKSUM,
#     PLUGIN_COMPILER_UBUNTU_24_04_CHECKSUM
# 
# The script expects the archive to contain:
#     build_manifest.json file with build information about the prebuilt compiler.
#         If the file is present, its content will be printed in cmake output.
#     lib/npu_driver_compiler.dll for Windows or lib/libnpu_driver_compiler.so for Linux that will be copied to
#         the output directory and renamed to openvino_intel_npu_compiler.dll
#         or libopenvino_intel_npu_compiler.so respectively.

function(print_build_manifest extracted_file)
    if(NOT EXISTS "${extracted_file}")
        message(WARNING "Build manifest file '${extracted_file}' not found. Skipping build_manifest information printing for plugin compiler.")
        return()
    endif()
    file(READ "${extracted_file}" FILE_CONTENT)
    string(REGEX REPLACE "[{}\"']" "" FILE_CONTENT "${FILE_CONTENT}")
    message(STATUS "build_manifest.json for npu plugin compiler:\n${FILE_CONTENT}")
endfunction()

if(ENABLE_INTEL_NPU_COMPILER)
    message(STATUS "Resolving prebuilt NPU Plugin Compiler dependencies...")

    set(PLUGIN_COMPILER_VERSION_MAJOR 7)
    set(PLUGIN_COMPILER_VERSION_MINOR 6)
    set(PLUGIN_COMPILER_VERSION_PATCH 0)
    set(PLUGIN_COMPILER_COMMIT_SHA da3cc32)
    set(PLUGIN_COMPILER_WINDOWS_2022_CHECKSUM 265bda74dfe07260e9b30ca02ec011653a57aa2900b40fd1f5291352922c363c)
    set(PLUGIN_COMPILER_UBUNTU_22_04_CHECKSUM db1a967d14ce47af64fcd8c66ced0ae8de68923bfd82fb02236fffa0fc625c0d)
    set(PLUGIN_COMPILER_UBUNTU_24_04_CHECKSUM c7b6fd5798cfc256fa28e9a75f1e150b75fb34a8441a4fbc453315805073c5aa)

    set(PLUGIN_COMPILER_VERSION_UNDERSCORE "${PLUGIN_COMPILER_VERSION_MAJOR}_${PLUGIN_COMPILER_VERSION_MINOR}_${PLUGIN_COMPILER_VERSION_PATCH}")
    message(STATUS "The prebuilt compiler version is ${PLUGIN_COMPILER_VERSION_MAJOR}.${PLUGIN_COMPILER_VERSION_MINOR}.${PLUGIN_COMPILER_VERSION_PATCH}.${PLUGIN_COMPILER_COMMIT_SHA}")

    if(WIN32)
        set(OS_FAMILY "windows")
        set(OS_VERSION_UNDERSCORE "2022")
        set(OS_UPPERCASE "WINDOWS_${OS_VERSION_UNDERSCORE}")

        set(PLUGIN_COMPILER_PACKAGE_PREFIX "npu_compiler_vcl_windows_${OS_VERSION_UNDERSCORE}")
        set(PLUGIN_COMPILER_PACKAGE_EXT "zip")
        set(PLUGIN_COMPILER_ARCHIVE_TYPE "ARCHIVE_WIN")
        set(PLUGIN_COMPILER_LIB_OLD_NAME "npu_driver_compiler.dll")
        set(PLUGIN_COMPILER_LIB_NEW_NAME "openvino_intel_npu_compiler.dll")
    elseif(UNIX AND NOT APPLE)
        # Get the OS name and OS version
        execute_process(COMMAND lsb_release -is OUTPUT_VARIABLE OS_NAME OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND lsb_release -rs OUTPUT_VARIABLE OS_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

        if(OS_NAME STREQUAL "Ubuntu" AND (OS_VERSION STREQUAL "22.04" OR OS_VERSION STREQUAL "24.04"))
            set(OS_FAMILY "linux")
            string(REPLACE "." "_" OS_VERSION_UNDERSCORE ${OS_VERSION})
            set(OS_UPPERCASE "UBUNTU_${OS_VERSION_UNDERSCORE}")

            set(PLUGIN_COMPILER_PACKAGE_PREFIX "npu_compiler_vcl_ubuntu_${OS_VERSION_UNDERSCORE}")
            set(PLUGIN_COMPILER_PACKAGE_EXT "tar.gz")
            set(PLUGIN_COMPILER_ARCHIVE_TYPE "ARCHIVE_LIN")
            set(PLUGIN_COMPILER_LIB_OLD_NAME "libnpu_driver_compiler.so")
            set(PLUGIN_COMPILER_LIB_NEW_NAME "libopenvino_intel_npu_compiler.so")
        else()
            message(STATUS "${OS_NAME} ${OS_VERSION} Linux distribution is not supported, skip downloading prebuilt Plugin Compiler libraries. Can not use plugin compiler libraries!")
            return()
        endif()
    else()
        message(STATUS "Current OS is not supported, skip downloading prebuilt Plugin Compiler libraries. Can not use plugin compiler libraries!")
        return()
    endif()

    set(PLUGIN_COMPILER_PACKAGE_SUBDIR "")
    set(PLUGIN_COMPILER_PACKAGE_NAME "${PLUGIN_COMPILER_PACKAGE_PREFIX}-${PLUGIN_COMPILER_VERSION_UNDERSCORE}-${PLUGIN_COMPILER_COMMIT_SHA}.${PLUGIN_COMPILER_PACKAGE_EXT}")
    if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
        set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
        set(PLUGIN_COMPILER_PACKAGE_SUBDIR "npu_compiler/")
    elseif(DEFINED THIRDPARTY_SERVER_PATH)
        set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
        set(PLUGIN_COMPILER_PACKAGE_SUBDIR "npu_compiler/")
    endif()

    RESOLVE_DEPENDENCY(NPU_PLUGIN_COMPILER
            ${PLUGIN_COMPILER_ARCHIVE_TYPE} "${PLUGIN_COMPILER_PACKAGE_SUBDIR}${PLUGIN_COMPILER_PACKAGE_NAME}"
            TARGET_PATH "${TEMP}/${PLATFORM_SUBDIR}/npu_compiler_${PLUGIN_COMPILER_VERSION_UNDERSCORE}_${PLUGIN_COMPILER_COMMIT_SHA}"
            ENVIRONMENT "NPU_PLUGIN_COMPILER_ROOT"
            FOLDER
            SHA256 "${PLUGIN_COMPILER_${OS_UPPERCASE}_CHECKSUM}"
            USE_NEW_LOCATION TRUE)

    if(NPU_PLUGIN_COMPILER)
        message(STATUS "Using prebuilt NPU Plugin Compiler libraries from ${NPU_PLUGIN_COMPILER}")
        print_build_manifest("${NPU_PLUGIN_COMPILER}/build_manifest.json")

        set(PLUGIN_COMPILER_LIB_PATH "${NPU_PLUGIN_COMPILER}/lib")

        configure_file(
            "${PLUGIN_COMPILER_LIB_PATH}/${PLUGIN_COMPILER_LIB_OLD_NAME}"
            "${PLUGIN_COMPILER_LIB_PATH}/${PLUGIN_COMPILER_LIB_NEW_NAME}"
            COPYONLY
        )

        if(USE_BUILD_TYPE_SUBFOLDER)
            set(PLUGIN_COMPILER_LIB_DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
        else()
            set(PLUGIN_COMPILER_LIB_DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}")
        endif()

        set(PLUGIN_COMPILER_LIB "${PLUGIN_COMPILER_LIB_PATH}/${PLUGIN_COMPILER_LIB_NEW_NAME}")
        file(COPY "${PLUGIN_COMPILER_LIB}" DESTINATION "${PLUGIN_COMPILER_LIB_DESTINATION}")
        message(STATUS "Copying prebuilt Plugin compiler library ${PLUGIN_COMPILER_LIB_PATH}/${PLUGIN_COMPILER_LIB_NEW_NAME} to ${PLUGIN_COMPILER_LIB_DESTINATION}")

        install(FILES ${PLUGIN_COMPILER_LIB} DESTINATION ${OV_CPACK_PLUGINSDIR} COMPONENT ${NPU_PLUGIN_COMPONENT})
    else()
        message(FATAL_ERROR "Failed to download prebuilt NPU Plugin Compiler libraries. Can not use plugin compiler libraries!")
    endif()
endif()
