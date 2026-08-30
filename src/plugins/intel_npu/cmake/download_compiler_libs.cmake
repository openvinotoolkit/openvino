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
#         ubuntu26.04: npu_compiler_vcl_ubuntu_26_04-<compiler_version>-<compiler_commit_sha>.tar.gz
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
#         If the file is present, its content will be printed in cmake output and the npu_compiler_version
#         value is written to runtime/npu_compiler_version.txt of the installation package.
#     lib folder with the following libraries that will be copied to the output directory
#     and included in the installation package:
#         WINDOWS: openvino_intel_npu_compiler.dll, openvino_intel_npu_compiler_loader.dll, openvino_intel_npu_vm_runtime.dll
#         LINUX: libopenvino_intel_npu_compiler.so, libopenvino_intel_npu_compiler_loader.so, libopenvino_intel_npu_vm_runtime.so

function(print_build_manifest extracted_file)
    if(NOT EXISTS "${extracted_file}")
        message(WARNING "Build manifest file '${extracted_file}' not found. Skipping build_manifest information printing for plugin compiler.")
        return()
    endif()
    file(READ "${extracted_file}" FILE_CONTENT)
    string(REGEX REPLACE "[{}\"']" "" FILE_CONTENT "${FILE_CONTENT}")
    message(STATUS "build_manifest.json for npu plugin compiler:\n${FILE_CONTENT}")
endfunction()

# Extracts npu_compiler_version from build_manifest.json and stores it in a standalone text file,
# so that the compiler version can be tracked in distribution archives without an API call
function(write_compiler_version_file manifest_file version_file)
    file(REMOVE "${version_file}")
    if(NOT EXISTS "${manifest_file}")
        message(WARNING "Build manifest file '${manifest_file}' not found. Skipping '${version_file}' generation.")
        return()
    endif()
    file(READ "${manifest_file}" MANIFEST_CONTENT)
    string(JSON COMPILER_VERSION ERROR_VARIABLE JSON_ERROR GET "${MANIFEST_CONTENT}" npu_compiler_version)
    if(JSON_ERROR)
        message(WARNING "Failed to read 'npu_compiler_version' from '${manifest_file}': ${JSON_ERROR}. Skipping '${version_file}' generation.")
        return()
    endif()
    file(WRITE "${version_file}" "${COMPILER_VERSION}\n")
    message(STATUS "Generated NPU compiler version file ${version_file} with version ${COMPILER_VERSION}")
endfunction()

if(ENABLE_INTEL_NPU_COMPILER)
    message(STATUS "Resolving prebuilt NPU Plugin Compiler dependencies...")

    set(PLUGIN_COMPILER_VERSION_MAJOR 8)
    set(PLUGIN_COMPILER_VERSION_MINOR 4)
    set(PLUGIN_COMPILER_VERSION_PATCH 0)
    set(PLUGIN_COMPILER_COMMIT_SHA ce3fa69)
    set(PLUGIN_COMPILER_WINDOWS_2022_CHECKSUM 5e9b416d9b5bdd06d7e6ad9cff00e673dad5ca60bfd29aae72bf23a8a5c40f42)
    set(PLUGIN_COMPILER_UBUNTU_22_04_CHECKSUM b7c3a3fa02fce1597c69687ccd3f919d26ba3550f31ea3d7050c9a66b7c65908)
    set(PLUGIN_COMPILER_UBUNTU_24_04_CHECKSUM b9b82f9ce441991439003535a16e600bd5eb9ef4e5fa41f4ad5cb526171c3de1)
    set(PLUGIN_COMPILER_UBUNTU_26_04_CHECKSUM e73f7ecbd49e747de47de273e141a1ac3e11c5b3c4a1ec4941b16a5dd33c65d5)

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
        set(PLUGIN_COMPILER_VM_RT_LIB_NAME "openvino_intel_npu_vm_runtime.dll")
        set(PLUGIN_COMPILER_VM_RT_PDB_NAME "openvino_intel_npu_vm_runtime.pdb")
    elseif(UNIX AND NOT APPLE AND NOT ANDROID)
        # Get the OS name and OS version
        execute_process(COMMAND lsb_release -is OUTPUT_VARIABLE OS_NAME OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND lsb_release -rs OUTPUT_VARIABLE OS_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

        if(OS_NAME STREQUAL "Ubuntu" AND (OS_VERSION STREQUAL "22.04" OR OS_VERSION STREQUAL "24.04" OR OS_VERSION STREQUAL "26.04"))
            set(OS_FAMILY "linux")
            string(REPLACE "." "_" OS_VERSION_UNDERSCORE ${OS_VERSION})
            set(OS_UPPERCASE "UBUNTU_${OS_VERSION_UNDERSCORE}")

            set(PLUGIN_COMPILER_PACKAGE_PREFIX "npu_compiler_vcl_ubuntu_${OS_VERSION_UNDERSCORE}")
            set(PLUGIN_COMPILER_PACKAGE_EXT "tar.gz")
            set(PLUGIN_COMPILER_ARCHIVE_TYPE "ARCHIVE_LIN")
            set(PLUGIN_COMPILER_LIB_NAME "libopenvino_intel_npu_compiler.so")
            set(PLUGIN_COMPILER_LOADER_LIB_NAME "libopenvino_intel_npu_compiler_loader.so")
            set(PLUGIN_COMPILER_VM_RT_LIB_NAME "libopenvino_intel_npu_vm_runtime.so")
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

        set(NPU_COMPILER_VERSION_FILE "${CMAKE_CURRENT_BINARY_DIR}/npu_compiler_version.txt")
        print_build_manifest("${NPU_PLUGIN_COMPILER}/build_manifest.json")
        write_compiler_version_file("${NPU_PLUGIN_COMPILER}/build_manifest.json" "${NPU_COMPILER_VERSION_FILE}")

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
        file(COPY "${PLUGIN_COMPILER_LIB}" DESTINATION "${PLUGIN_COMPILER_LIB_DESTINATION}")
        file(COPY "${PLUGIN_COMPILER_LOADER_LIB}" DESTINATION "${PLUGIN_COMPILER_LIB_DESTINATION}")
        file(COPY "${PLUGIN_COMPILER_VM_RT_LIB}" DESTINATION "${PLUGIN_COMPILER_LIB_DESTINATION}")
        message(STATUS "Copying prebuilt Plugin compiler library ${PLUGIN_COMPILER_LIB} to ${PLUGIN_COMPILER_LIB_DESTINATION}")
        message(STATUS "Copying prebuilt Plugin compiler loader library ${PLUGIN_COMPILER_LOADER_LIB} to ${PLUGIN_COMPILER_LIB_DESTINATION}")
        message(STATUS "Copying prebuilt Plugin compiler VM runtime library ${PLUGIN_COMPILER_VM_RT_LIB} to ${PLUGIN_COMPILER_LIB_DESTINATION}")

        install(FILES ${PLUGIN_COMPILER_LIB} DESTINATION ${OV_CPACK_PLUGINSDIR} COMPONENT ${NPU_PLUGIN_COMPONENT})
        install(FILES ${PLUGIN_COMPILER_LOADER_LIB} DESTINATION ${OV_CPACK_PLUGINSDIR} COMPONENT ${NPU_PLUGIN_COMPONENT})
        if(ENABLE_TESTS)
            install(FILES ${PLUGIN_COMPILER_LIB} DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
            install(FILES ${PLUGIN_COMPILER_LOADER_LIB} DESTINATION tests COMPONENT tests EXCLUDE_FROM_ALL)
        endif()
        install(FILES ${PLUGIN_COMPILER_VM_RT_LIB} DESTINATION ${OV_CPACK_PLUGINSDIR} COMPONENT ${NPU_PLUGIN_COMPONENT})

        if(EXISTS "${NPU_COMPILER_VERSION_FILE}")
            install(FILES ${NPU_COMPILER_VERSION_FILE} DESTINATION runtime COMPONENT ${NPU_PLUGIN_COMPONENT})
        endif()

        if(WIN32)
            set(PLUGIN_COMPILER_PDB "${PLUGIN_COMPILER_PDB_PATH}/${PLUGIN_COMPILER_PDB_NAME}")
            set(PLUGIN_COMPILER_LOADER_PDB "${PLUGIN_COMPILER_PDB_PATH}/${PLUGIN_COMPILER_LOADER_PDB_NAME}")
            set(PLUGIN_COMPILER_VM_RT_PDB "${PLUGIN_COMPILER_PDB_PATH}/${PLUGIN_COMPILER_VM_RT_PDB_NAME}")
            file(COPY "${PLUGIN_COMPILER_PDB}" DESTINATION "${PLUGIN_COMPILER_LIB_DESTINATION}")
            file(COPY "${PLUGIN_COMPILER_LOADER_PDB}" DESTINATION "${PLUGIN_COMPILER_LIB_DESTINATION}")
            file(COPY "${PLUGIN_COMPILER_VM_RT_PDB}" DESTINATION "${PLUGIN_COMPILER_LIB_DESTINATION}")
            message(STATUS "Copying prebuilt Plugin compiler PDB files from ${PLUGIN_COMPILER_PDB_PATH} to ${PLUGIN_COMPILER_LIB_DESTINATION}")

            install(FILES ${PLUGIN_COMPILER_PDB} DESTINATION ${OV_CPACK_PLUGINSDIR} COMPONENT pdb EXCLUDE_FROM_ALL)
            install(FILES ${PLUGIN_COMPILER_LOADER_PDB} DESTINATION ${OV_CPACK_PLUGINSDIR} COMPONENT pdb EXCLUDE_FROM_ALL)
            install(FILES ${PLUGIN_COMPILER_VM_RT_PDB} DESTINATION ${OV_CPACK_PLUGINSDIR} COMPONENT pdb EXCLUDE_FROM_ALL)
        endif()
    else()
        message(FATAL_ERROR "Failed to download prebuilt NPU Plugin Compiler libraries. Can not use plugin compiler libraries!")
    endif()
endif()
 
