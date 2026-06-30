# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# This script resolves the prebuilt NPU Plugin Compiler dependency by downloading and extracting the appropriate
# archive based on the current platform. The expected location of the archive and naming convention is as follows:
#     storage location: https://storage.openvinotoolkit.org/dependencies/thirdparty
#     WINDOWS: 
#         windows2022: npu_compiler_vcl_windows_2022-<compiler_version>-<compiler_commit_sha>.zip
#     LINUX:
#         ubuntu22.04: npu_compiler_vcl_ubuntu_22_04-<compiler_version>-<compiler_commit_sha>.tar.gz
#         ubuntu24.04: npu_compiler_vcl_ubuntu_24_04-<compiler_version>-<compiler_commit_sha>.tar.gz
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
#     lib folder with the following libraries that will be copied to the output directory
#     and included in the installation package:
#         WINDOWS: openvino_intel_npu_compiler.dll, openvino_intel_npu_compiler_loader.dll
#         LINUX: libopenvino_intel_npu_compiler.so, libopenvino_intel_npu_compiler_loader.so

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

    set(PLUGIN_COMPILER_VERSION_MAJOR 8)
    set(PLUGIN_COMPILER_VERSION_MINOR 2)
    set(PLUGIN_COMPILER_VERSION_PATCH 0)
    set(PLUGIN_COMPILER_COMMIT_SHA 04eb7b8)
    set(PLUGIN_COMPILER_WINDOWS_2022_CHECKSUM 7177f86848af215b11d02de4a617ac71222bea2c6ff298531af3346cf488cff0)
    set(PLUGIN_COMPILER_UBUNTU_22_04_CHECKSUM 61fbb48ca069e0ebb5b9ce3b9959fc38b08795a78f42b2ae334487da8c43e3f3)
    set(PLUGIN_COMPILER_UBUNTU_24_04_CHECKSUM 88a76e0ea6502952e7abbd15cf791f8731411f7c1de93c194182a08732eb08eb)

    set(PLUGIN_COMPILER_VERSION_UNDERSCORE "${PLUGIN_COMPILER_VERSION_MAJOR}_${PLUGIN_COMPILER_VERSION_MINOR}_${PLUGIN_COMPILER_VERSION_PATCH}")
    message(STATUS "The prebuilt compiler version is ${PLUGIN_COMPILER_VERSION_MAJOR}.${PLUGIN_COMPILER_VERSION_MINOR}.${PLUGIN_COMPILER_VERSION_PATCH}.${PLUGIN_COMPILER_COMMIT_SHA}")

    if(WIN32)
        set(OS_FAMILY "windows")
        set(OS_VERSION_UNDERSCORE "2022")
        set(OS_UPPERCASE "WINDOWS_${OS_VERSION_UNDERSCORE}")

        set(PLUGIN_COMPILER_PACKAGE_PREFIX "npu_compiler_vcl_windows_${OS_VERSION_UNDERSCORE}")
        set(PLUGIN_COMPILER_PACKAGE_EXT "zip")
        set(PLUGIN_COMPILER_ARCHIVE_TYPE "ARCHIVE_WIN")
        set(PLUGIN_COMPILER_LIB_NAME "openvino_intel_npu_compiler.dll")
        set(PLUGIN_COMPILER_PDB_NAME "openvino_intel_npu_compiler.pdb")
        set(PLUGIN_COMPILER_LOADER_LIB_NAME "openvino_intel_npu_compiler_loader.dll")
        set(PLUGIN_COMPILER_LOADER_PDB_NAME "openvino_intel_npu_compiler_loader.pdb")
        set(PLUGIN_COMPILER_VM_RT_LIB_NAME "npu_interpreter_runtime.dll")
        set(PLUGIN_COMPILER_VM_RT_RENAMED_LIB_NAME "openvino_intel_npu_vm_runtime.dll")
        set(PLUGIN_COMPILER_VM_RT_PDB_NAME "npu_interpreter_runtime.pdb")
        set(PLUGIN_COMPILER_VM_RT_RENAMED_PDB_NAME "openvino_intel_npu_vm_runtime.pdb")
    elseif(UNIX AND NOT APPLE AND NOT ANDROID)
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
            set(PLUGIN_COMPILER_LIB_NAME "libopenvino_intel_npu_compiler.so")
            set(PLUGIN_COMPILER_LOADER_LIB_NAME "libopenvino_intel_npu_compiler_loader.so")
            set(PLUGIN_COMPILER_VM_RT_LIB_NAME "libnpu_interpreter_runtime.so")
            set(PLUGIN_COMPILER_VM_RT_RENAMED_LIB_NAME "libopenvino_intel_npu_vm_runtime.so")
        else()
            message(STATUS "${OS_NAME} ${OS_VERSION} Linux distribution is not supported, skip downloading prebuilt Plugin Compiler libraries. Can not use plugin compiler libraries!")
            return()
        endif()
    else()
        message(STATUS "Current OS is not supported, skip downloading prebuilt Plugin Compiler libraries. Can not use plugin compiler libraries!")
        return()
    endif()

    set(PLUGIN_COMPILER_PACKAGE_NAME "${PLUGIN_COMPILER_PACKAGE_PREFIX}-${PLUGIN_COMPILER_VERSION_UNDERSCORE}-${PLUGIN_COMPILER_COMMIT_SHA}.${PLUGIN_COMPILER_PACKAGE_EXT}")
    if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
        set(IE_PATH_TO_DEPS "$ENV{THIRDPARTY_SERVER_PATH}")
    elseif(DEFINED THIRDPARTY_SERVER_PATH)
        set(IE_PATH_TO_DEPS "${THIRDPARTY_SERVER_PATH}")
    endif()

    RESOLVE_DEPENDENCY(NPU_PLUGIN_COMPILER
            ${PLUGIN_COMPILER_ARCHIVE_TYPE} "npu_compiler/${PLUGIN_COMPILER_PACKAGE_NAME}"
            TARGET_PATH "${TEMP}/${PLATFORM_SUBDIR}/npu_compiler_${PLUGIN_COMPILER_VERSION_UNDERSCORE}_${PLUGIN_COMPILER_COMMIT_SHA}"
            ENVIRONMENT "NPU_PLUGIN_COMPILER_ROOT"
            FOLDER
            SHA256 "${PLUGIN_COMPILER_${OS_UPPERCASE}_CHECKSUM}"
            USE_NEW_LOCATION TRUE)

    if(NPU_PLUGIN_COMPILER)
        message(STATUS "Using prebuilt NPU Plugin Compiler libraries from ${NPU_PLUGIN_COMPILER}")
        print_build_manifest("${NPU_PLUGIN_COMPILER}/build_manifest.json")

        set(PLUGIN_COMPILER_LIB_PATH "${NPU_PLUGIN_COMPILER}/lib")
        set(PLUGIN_COMPILER_PDB_PATH "${NPU_PLUGIN_COMPILER}/pdb")

        if(USE_BUILD_TYPE_SUBFOLDER)
            set(PLUGIN_COMPILER_LIB_DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
        else()
            set(PLUGIN_COMPILER_LIB_DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}")
        endif()

        set(PLUGIN_COMPILER_LIB "${PLUGIN_COMPILER_LIB_PATH}/${PLUGIN_COMPILER_LIB_NAME}")
        set(PLUGIN_COMPILER_LOADER_LIB "${PLUGIN_COMPILER_LIB_PATH}/${PLUGIN_COMPILER_LOADER_LIB_NAME}")
        set(PLUGIN_COMPILER_VM_RT_LIB "${PLUGIN_COMPILER_LIB_PATH}/${PLUGIN_COMPILER_VM_RT_LIB_NAME}")
        set(PLUGIN_COMPILER_VM_RT_RENAMED_LIB "${PLUGIN_COMPILER_LIB_DESTINATION}/${PLUGIN_COMPILER_VM_RT_RENAMED_LIB_NAME}")
        file(COPY "${PLUGIN_COMPILER_LIB}" DESTINATION "${PLUGIN_COMPILER_LIB_DESTINATION}")
        file(COPY "${PLUGIN_COMPILER_LOADER_LIB}" DESTINATION "${PLUGIN_COMPILER_LIB_DESTINATION}")
        configure_file("${PLUGIN_COMPILER_VM_RT_LIB}" "${PLUGIN_COMPILER_VM_RT_RENAMED_LIB}" COPYONLY)
        message(STATUS "Copying prebuilt Plugin compiler library ${PLUGIN_COMPILER_LIB} to ${PLUGIN_COMPILER_LIB_DESTINATION}")
        message(STATUS "Copying prebuilt Plugin compiler loader library ${PLUGIN_COMPILER_LOADER_LIB} to ${PLUGIN_COMPILER_LIB_DESTINATION}")
        message(STATUS "Copying prebuilt Plugin compiler VM runtime library ${PLUGIN_COMPILER_VM_RT_LIB} to ${PLUGIN_COMPILER_VM_RT_RENAMED_LIB}")

        install(FILES ${PLUGIN_COMPILER_LIB} DESTINATION ${OV_CPACK_PLUGINSDIR} COMPONENT ${NPU_PLUGIN_COMPONENT})
        install(FILES ${PLUGIN_COMPILER_LOADER_LIB} DESTINATION ${OV_CPACK_PLUGINSDIR} COMPONENT ${NPU_PLUGIN_COMPONENT})
        if(ENABLE_INTEL_NPU_INTERNAL)
            install(FILES ${PLUGIN_COMPILER_VM_RT_RENAMED_LIB} DESTINATION ${OV_CPACK_PLUGINSDIR} COMPONENT ${NPU_INTERNAL_COMPONENT} ${OV_CPACK_COMP_NPU_INTERNAL_EXCLUDE_ALL})
        endif()

        if(WIN32)
            set(PLUGIN_COMPILER_PDB "${PLUGIN_COMPILER_PDB_PATH}/${PLUGIN_COMPILER_PDB_NAME}")
            set(PLUGIN_COMPILER_LOADER_PDB "${PLUGIN_COMPILER_PDB_PATH}/${PLUGIN_COMPILER_LOADER_PDB_NAME}")
            set(PLUGIN_COMPILER_VM_RT_PDB "${PLUGIN_COMPILER_PDB_PATH}/${PLUGIN_COMPILER_VM_RT_PDB_NAME}")
            set(PLUGIN_COMPILER_VM_RT_RENAMED_PDB "${PLUGIN_COMPILER_LIB_DESTINATION}/${PLUGIN_COMPILER_VM_RT_RENAMED_PDB_NAME}")
            file(COPY "${PLUGIN_COMPILER_PDB}" DESTINATION "${PLUGIN_COMPILER_LIB_DESTINATION}")
            file(COPY "${PLUGIN_COMPILER_LOADER_PDB}" DESTINATION "${PLUGIN_COMPILER_LIB_DESTINATION}")
            configure_file("${PLUGIN_COMPILER_VM_RT_PDB}" "${PLUGIN_COMPILER_VM_RT_RENAMED_PDB}" COPYONLY)
            message(STATUS "Copying prebuilt Plugin compiler PDB files from ${PLUGIN_COMPILER_PDB_PATH} to ${PLUGIN_COMPILER_LIB_DESTINATION}")

            install(FILES ${PLUGIN_COMPILER_PDB} DESTINATION ${OV_CPACK_PLUGINSDIR} COMPONENT pdb EXCLUDE_FROM_ALL)
            install(FILES ${PLUGIN_COMPILER_LOADER_PDB} DESTINATION ${OV_CPACK_PLUGINSDIR} COMPONENT pdb EXCLUDE_FROM_ALL)
        endif()
    else()
        message(FATAL_ERROR "Failed to download prebuilt NPU Plugin Compiler libraries. Can not use plugin compiler libraries!")
    endif()
endif()
