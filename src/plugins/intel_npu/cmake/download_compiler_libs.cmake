# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# This CMake module is responsible for downloading and extracting the prebuilt Intel NPU Plugin Compiler libraries,
# which are required for the NPU plugin to function. It defines functions to handle the download and extraction process,
# including checksum verification and error handling.
# The module tries to download packages from 3 different locations in order of preference, and will fallback to the next one if the download fails.
#     Attempt #1: Latest Compiler from the public storage.
#     Attempt #2: Latest Compiler from the custom server (if environment variable THIRDPARTY_SERVER_PATH is set).
#     Attempt #3: Fallback Compiler from the public storage.

function(download_npu_plugin_compiler_libs
    urls         # list of URLs to try downloading the prebuilt Plugin Compiler libraries from, in order of preference
    checksums    # list of corresponding SHA256 checksums for the URLs, in the same order
    parent_dir   # directory where the prebuilt Plugin Compiler libraries should be downloaded and extracted
    result_path  # output variable to hold the path of the downloaded Plugin Compiler libraries archive
)
    # Check if the prebuilt Plugin Compiler archive already exists.
    foreach(url IN LISTS urls)
        get_filename_component(archive_name "${url}" NAME)
        if(EXISTS "${parent_dir}/${archive_name}")
            message(STATUS "Prebuilt Plugin Compiler libraries already downloaded ${parent_dir}/${archive_name}, skip download")
            set(${result_path} "${parent_dir}/${archive_name}" PARENT_SCOPE)
            return()
        endif()
        get_filename_component(archive_extracted_name "${url}" NAME_WE)
        if(EXISTS "${parent_dir}/${archive_extracted_name}")
            message(STATUS "Prebuilt Plugin Compiler libraries already extracted ${parent_dir}/${archive_extracted_name}, skip extraction")
            set(${result_path} "${parent_dir}/${archive_name}" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    # Try downloading from each URL in the list, fallback to next on failure
    list(LENGTH urls url_count)
    math(EXPR url_last_index "${url_count} - 1")
    foreach(index RANGE ${url_last_index})
        list(GET urls ${index} url)
        list(GET checksums ${index} checksum)
        if("${url}" STREQUAL "")
            continue()
        endif()
        # Derive archive name from URL (everything after last "/")
        get_filename_component(archive_name "${url}" NAME)
        get_filename_component(archive_name_no_ext "${archive_name}" NAME_WE)
        set(download_archive_path "${parent_dir}/${archive_name}")

        math(EXPR attempt_number "${index} + 1")
        message(STATUS "[Attempt #${attempt_number}] Downloading prebuilt Plugin Compiler libraries from ${url}")
        file(DOWNLOAD "${url}" "${download_archive_path}"
            TIMEOUT 3600
            LOG log_output
            STATUS download_status
            SHOW_PROGRESS)

        list(GET download_status 0 download_result)
        if(NOT download_result EQUAL 0)
            message(STATUS "Download failed from ${url}\nStatus: ${download_status}\nLog: ${log_output}")
            file(REMOVE "${download_archive_path}")
            continue()
        endif()

        # Verify checksum after successful download
        file(SHA256 "${download_archive_path}" actual_checksum)
        if(NOT "${actual_checksum}" STREQUAL "${checksum}")
            message(STATUS "Checksum mismatch for ${download_archive_path}\n\tExpected: ${checksum}\n\tActual:   ${actual_checksum}")
            file(REMOVE "${download_archive_path}")
            continue()
        endif()

        message(STATUS "Download completed and checksum verified: ${download_archive_path}")
        set(${result_path} "${download_archive_path}" PARENT_SCOPE)
        return()
    endforeach()
    message(FATAL_ERROR "All download URLs failed for prebuilt Plugin Compiler libraries. Please check the URLs or your network connection.")
endfunction()

function(extract_npu_plugin_compiler_libs
    download_archive_path  # path to the downloaded prebuilt Plugin Compiler libraries archive
    result_path            # output variable to hold the path of the extracted prebuilt Plugin Compiler libraries
)
    get_filename_component(parent_dir "${download_archive_path}" DIRECTORY)
    get_filename_component(download_archive_name_we "${download_archive_path}" NAME_WE)
    set(extracted_archive_dir "${parent_dir}/${download_archive_name_we}")

    # Check if the prebuilt Plugin Compiler directory already exists.
    if(EXISTS "${extracted_archive_dir}")
        set(${result_path} "${extracted_archive_dir}" PARENT_SCOPE)
        return()
    endif()

    message(STATUS "Unzipping prebuilt Plugin Compiler libraries to ${extracted_archive_dir}")

    # Determine extraction method based on file extension
    if("${download_archive_path}" MATCHES "\\.zip$")
        file(ARCHIVE_EXTRACT INPUT "${download_archive_path}" DESTINATION "${extracted_archive_dir}")
    elseif("${download_archive_path}" MATCHES "\\.tar.gz$")
        if(NOT EXISTS "${extracted_archive_dir}")
            file(MAKE_DIRECTORY "${extracted_archive_dir}")
            message(STATUS "Directory ${extracted_archive_dir} created to unzip.")
        endif()
        execute_process(COMMAND tar -xzf "${download_archive_path}" -C "${extracted_archive_dir}")
    elseif("${download_archive_path}" MATCHES "\\.deb$")
        execute_process(COMMAND dpkg-deb -x "${download_archive_path}" "${extracted_archive_dir}")
    elseif("${download_archive_path}" MATCHES "\\.exe$")
        set(WINRAR_PATHS
            "C:/Program Files/WinRAR"
            "C:/Program Files (x86)/WinRAR"
        )

        set(WINRAR_FOUND FALSE)
        set(WINRAR_EXECUTABLE "")

        foreach(PATH ${WINRAR_PATHS})
            if(EXISTS "${PATH}/WinRAR.exe")
                set(WINRAR_FOUND TRUE)
                set(WINRAR_EXECUTABLE "${PATH}/WinRAR.exe")
                break()
            endif()
        endforeach()

        if(WINRAR_FOUND)
            message(STATUS "WinRAR found at: ${WINRAR_EXECUTABLE} and extract ${download_archive_path} to ${extracted_archive_dir}")
            file(MAKE_DIRECTORY "${extracted_archive_dir}")
            execute_process(
                COMMAND "${WINRAR_EXECUTABLE}" x -y -o+ "${download_archive_path}" "${extracted_archive_dir}"
                RESULT_VARIABLE result
                OUTPUT_VARIABLE output 
                ERROR_VARIABLE error
            )

            if(result EQUAL 0)
                message(STATUS "Extraction successful: ${output}")
            else()
                #file(REMOVE_RECURSE "${extracted_archive_dir}")
                message(STATUS "Extraction failed: ${error}")
            endif()
        else()
            message(FATAL_ERROR "WinRAR not found. Please install WinRAR to proceed.")
        endif()
    else()
        message(FATAL_ERROR "Unsupported file extension for extraction: ${download_archive_path}")
    endif()
    file(REMOVE "${download_archive_path}")
    set(${result_path} "${extracted_archive_dir}" PARENT_SCOPE)
endfunction()

function(print_build_manifest extracted_file)
    if(NOT EXISTS "${extracted_file}")
        message(WARNING "Build manifest file '${extracted_file}' not found. Skipping build_manifest information printing for plugin compiler.")
        return()
    endif()
    # read the build_manifest.json file
    file(READ "${extracted_file}" FILE_CONTENT)
    string(REGEX REPLACE "[{}\"']" "" FILE_CONTENT "${FILE_CONTENT}")
    message(STATUS "build_manifest.json for npu plugin compiler:\n${FILE_CONTENT}")
endfunction()

if(ENABLE_INTEL_NPU_COMPILER)
    message(STATUS "Downloading prebuilt NPU Plugin compiler libraries")
    set(PLUGIN_COMPILER_VERSION_MAJOR_LATEST 7)
    set(PLUGIN_COMPILER_VERSION_MINOR_LATEST 6)
    set(PLUGIN_COMPILER_VERSION_PATCH_LATEST 0)
    set(PLUGIN_COMPILER_COMMIT_SHA_LATEST 45604ce)
    set(PLUGIN_COMPILER_WINDOWS_2022_CHECKSUM_LATEST 631f435078de14938c9452cbae166d1a7b62ec5f16d4cd2a7d7b4bdf986959fa)
    set(PLUGIN_COMPILER_UBUNTU_22_04_CHECKSUM_LATEST ed1d5bbc81871eceed99c4d679b2c2aa6efd3c74a6c99b363169dab7c22c3620)
    set(PLUGIN_COMPILER_UBUNTU_24_04_CHECKSUM_LATEST 6ae879ed113fa57994f7aa799b0d074591971cf7f7a3821b6305475f1616a6ee)
    set(PLUGIN_COMPILER_VERSION_LATEST "${PLUGIN_COMPILER_VERSION_MAJOR_LATEST}_${PLUGIN_COMPILER_VERSION_MINOR_LATEST}_${PLUGIN_COMPILER_VERSION_PATCH_LATEST}")

    set(PLUGIN_COMPILER_VERSION_MAJOR_FALLBACK 7)
    set(PLUGIN_COMPILER_VERSION_MINOR_FALLBACK 6)
    set(PLUGIN_COMPILER_VERSION_PATCH_FALLBACK 0)
    set(PLUGIN_COMPILER_COMMIT_SHA_FALLBACK da3cc32)
    set(PLUGIN_COMPILER_WINDOWS_2022_CHECKSUM_FALLBACK 265bda74dfe07260e9b30ca02ec011653a57aa2900b40fd1f5291352922c363c)
    set(PLUGIN_COMPILER_UBUNTU_22_04_CHECKSUM_FALLBACK db1a967d14ce47af64fcd8c66ced0ae8de68923bfd82fb02236fffa0fc625c0d)
    set(PLUGIN_COMPILER_UBUNTU_24_04_CHECKSUM_FALLBACK c7b6fd5798cfc256fa28e9a75f1e150b75fb34a8441a4fbc453315805073c5aa)
    set(PLUGIN_COMPILER_VERSION_FALLBACK "${PLUGIN_COMPILER_VERSION_MAJOR_FALLBACK}_${PLUGIN_COMPILER_VERSION_MINOR_FALLBACK}_${PLUGIN_COMPILER_VERSION_PATCH_FALLBACK}")

    message(STATUS "The prebuilt compiler version is ${PLUGIN_COMPILER_VERSION_MAJOR_LATEST}.${PLUGIN_COMPILER_VERSION_MINOR_LATEST}.${PLUGIN_COMPILER_VERSION_PATCH_LATEST}")

    set(URLS "")
    set(CHECKSUMS "")

    if(WIN32)
        set(OS_FAMILY "windows")
        set(OS_VERSION_UNDERSCORE "2022")
        set(OS_UPPERCASE "WINDOWS_${OS_VERSION_UNDERSCORE}")

        set(PLUGIN_COMPILER_PACKAGE_PREFIX "npu_compiler_vcl_windows_${OS_VERSION_UNDERSCORE}")
        set(PLUGIN_COMPILER_PACKAGE_EXT "zip")
        set(PLUGIN_COMPILER_LIB_OLD_NAME "npu_driver_compiler.dll")
        set(PLUGIN_COMPILER_LIB_NEW_NAME "openvino_intel_npu_compiler.dll")
    elseif(UNIX AND NOT APPLE)
        # Get the OS name and version
        execute_process(COMMAND lsb_release -is OUTPUT_VARIABLE OS_NAME OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND lsb_release -rs OUTPUT_VARIABLE OS_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

        if(OS_NAME STREQUAL "Ubuntu" AND (OS_VERSION STREQUAL "22.04" OR OS_VERSION STREQUAL "24.04"))
            set(OS_FAMILY "linux")
            string(REPLACE "." "_" OS_VERSION_UNDERSCORE ${OS_VERSION})
            set(OS_UPPERCASE "UBUNTU_${OS_VERSION_UNDERSCORE}")

            set(PLUGIN_COMPILER_PACKAGE_PREFIX "npu_compiler_vcl_ubuntu_${OS_VERSION_UNDERSCORE}")
            set(PLUGIN_COMPILER_PACKAGE_EXT "tar.gz")
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

    set(PLUGIN_COMPILER_BASE_PUBLIC_URL "https://storage.openvinotoolkit.org/dependencies")

    #
    # Option 1: Latest Compiler from the public storage.
    #

    list(APPEND URLS "${PLUGIN_COMPILER_BASE_PUBLIC_URL}/thirdparty/${OS_FAMILY}/${PLUGIN_COMPILER_PACKAGE_PREFIX}-${PLUGIN_COMPILER_VERSION_LATEST}-${PLUGIN_COMPILER_COMMIT_SHA_LATEST}.${PLUGIN_COMPILER_PACKAGE_EXT}")
    list(APPEND CHECKSUMS ${PLUGIN_COMPILER_${OS_UPPERCASE}_CHECKSUM_LATEST})

    #
    # Option 2: Latest Compiler from the custom server.
    #

    if(DEFINED ENV{THIRDPARTY_SERVER_PATH})
        list(APPEND URLS "$ENV{THIRDPARTY_SERVER_PATH}/thirdparty/${OS_FAMILY}/npu_compiler/${PLUGIN_COMPILER_PACKAGE_PREFIX}-${PLUGIN_COMPILER_VERSION_LATEST}-${PLUGIN_COMPILER_COMMIT_SHA_LATEST}.${PLUGIN_COMPILER_PACKAGE_EXT}")
        list(APPEND CHECKSUMS ${PLUGIN_COMPILER_${OS_UPPERCASE}_CHECKSUM_LATEST})
    elseif(DEFINED THIRDPARTY_SERVER_PATH)
        list(APPEND URLS "${THIRDPARTY_SERVER_PATH}/thirdparty/${OS_FAMILY}/npu_compiler/${PLUGIN_COMPILER_PACKAGE_PREFIX}-${PLUGIN_COMPILER_VERSION_LATEST}-${PLUGIN_COMPILER_COMMIT_SHA_LATEST}.${PLUGIN_COMPILER_PACKAGE_EXT}")
        list(APPEND CHECKSUMS ${PLUGIN_COMPILER_${OS_UPPERCASE}_CHECKSUM_LATEST})
    endif()

    #
    # Option 3: Fallback Compiler from the public storage.
    #

    list(APPEND URLS "${PLUGIN_COMPILER_BASE_PUBLIC_URL}/thirdparty/${OS_FAMILY}/${PLUGIN_COMPILER_PACKAGE_PREFIX}-${PLUGIN_COMPILER_VERSION_FALLBACK}-${PLUGIN_COMPILER_COMMIT_SHA_FALLBACK}.${PLUGIN_COMPILER_PACKAGE_EXT}")
    list(APPEND CHECKSUMS ${PLUGIN_COMPILER_${OS_UPPERCASE}_CHECKSUM_FALLBACK})

    #
    # Downloading and extracting the prebuilt Plugin Compiler libraries
    #

    set(PLUGIN_COMPILER_PARENT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/temp/plugin_compiler_lib/${OS_FAMILY}")
    download_npu_plugin_compiler_libs("${URLS}" "${CHECKSUMS}" "${PLUGIN_COMPILER_PARENT_DIR}" PLUGIN_COMPILER_ARCHIVE_PATH)
    extract_npu_plugin_compiler_libs("${PLUGIN_COMPILER_ARCHIVE_PATH}" PLUGIN_COMPILER_ARCHIVE_EXTRACTED_DIR)
    print_build_manifest("${PLUGIN_COMPILER_ARCHIVE_EXTRACTED_DIR}/build_manifest.json")

    #
    # Renaming and copying the prebuilt Plugin Compiler libraries to the destination folder for NPU plugin
    #

    set(PLUGIN_COMPILER_LIB_PATH "${PLUGIN_COMPILER_ARCHIVE_EXTRACTED_DIR}/lib")
    configure_file(
        ${PLUGIN_COMPILER_LIB_PATH}/${PLUGIN_COMPILER_LIB_OLD_NAME}
        ${PLUGIN_COMPILER_LIB_PATH}/${PLUGIN_COMPILER_LIB_NEW_NAME}
        COPYONLY
    )

    # The destinations are the same. CMAKE_BUILD_TYPE is added based on the option USE_BUILD_TYPE_SUBFOLDER
    if(USE_BUILD_TYPE_SUBFOLDER)
        set(NPU_COMPILER_LIB_DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
    else()
        set(NPU_COMPILER_LIB_DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}")
    endif()
    file(COPY "${PLUGIN_COMPILER_LIB_PATH}/${PLUGIN_COMPILER_LIB_NEW_NAME}" DESTINATION "${NPU_COMPILER_LIB_DESTINATION}")
    message(STATUS "Copying prebuilt Plugin Compiler libraries ${PLUGIN_COMPILER_LIB_PATH}/${PLUGIN_COMPILER_LIB_NEW_NAME} to ${NPU_COMPILER_LIB_DESTINATION}")

    install(FILES "${PLUGIN_COMPILER_LIB_PATH}/${PLUGIN_COMPILER_LIB_NEW_NAME}" DESTINATION ${OV_CPACK_PLUGINSDIR} COMPONENT ${NPU_PLUGIN_COMPONENT})
endif()
