# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Function to download and extract files
function(download_and_extract url zip_file extracted_dir modify_proxy)
    # Check if the prebuilt Plugin compiler libraries not exist
    if(NOT EXISTS "${extracted_dir}")
        # Download the prebuilt Plugin compiler libraries, if failure, show error message and exit
        if(NOT "${url}" STREQUAL "")
            if(modify_proxy STREQUAL "MODIFY")
                # Update proxy to enable download for windows url
            set(original_NO_PROXY $ENV{NO_PROXY})
                set(original_no_proxy $ENV{no_proxy})
                set(ENV{NO_PROXY} "")
                set(ENV{no_proxy} "")
            endif()
            
            message(STATUS "${url} is not empty")
            message(STATUS "Downloading prebuilt Plugin compiler libraries from ${url}")
            file(DOWNLOAD "${url}" "${zip_file}"
                TIMEOUT 3600
                LOG log_output
                STATUS download_status
                SHOW_PROGRESS)

            if(modify_proxy STREQUAL "MODIFY")
                # Restore proxy
                set(ENV{NO_PROXY} ${original_NO_PROXY})
                set(ENV{no_proxy} ${original_no_proxy})
            endif()

            list(GET download_status 0 download_result)
            if(NOT download_result EQUAL 0)
                message(FATAL_ERROR "Download failed!\nStatus: ${download_status}\nLog: ${log_output}")
            else()
                message(STATUS "Download completed: ${zip_file}")
            endif()
        endif()

        message(STATUS "Unzipping prebuilt Plugin compiler libraries to ${extracted_dir}")
        # Determine extraction method based on file extension
        if("${zip_file}" MATCHES "\\.zip$")
            file(ARCHIVE_EXTRACT INPUT "${zip_file}" DESTINATION "${extracted_dir}")
        elseif("${zip_file}" MATCHES "\\.tar.gz$")
            if(NOT EXISTS "${extracted_dir}")
                file(MAKE_DIRECTORY "${extracted_dir}")
                message(STATUS "Directory ${extracted_dir} created to unzip.")
            endif()
            execute_process(COMMAND tar -xzf "${zip_file}" -C "${extracted_dir}")
        elseif("${zip_file}" MATCHES "\\.deb$")
            execute_process(COMMAND dpkg-deb -x "${zip_file}" "${extracted_dir}")
        elseif("${zip_file}" MATCHES "\\.exe$")
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
                message(STATUS "WinRAR found at: ${WINRAR_EXECUTABLE} and extract ${zip_file} to ${extracted_dir}")
                file(MAKE_DIRECTORY "${extracted_dir}")
                execute_process(
                    COMMAND "${WINRAR_EXECUTABLE}" x -y -o+ "${zip_file}" "${extracted_dir}"
                    RESULT_VARIABLE result
                    OUTPUT_VARIABLE output 
                    ERROR_VARIABLE error
                )

                if(result EQUAL 0)
                    message(STATUS "Extraction successful: ${output}")
                else()
                    #file(REMOVE_RECURSE "${extracted_dir}")
                    message(STATUS "Extraction failed: ${error}")
                endif()
            else()
                message(FATAL_ERROR "WinRAR not found. Please install WinRAR to proceed.")
            endif()
        else()
            message(FATAL_ERROR "Unsupported file extension for extraction: ${zip_file}")
        endif()
        file(REMOVE "${zip_file}")
    else()
        message(STATUS "Prebuilt Plugin compiler libraries already exist, skip download")
    endif()
endfunction()

if(ENABLE_PLUGIN_COMPILER_DOWNLOAD)
    message(STATUS "Downloading prebuilt NPU Plugin compiler libraries")
    if(WIN32)
        set(PLUGIN_COMPILER_LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/temp/plugin_compiler_lib/win")
        set(PLUGIN_COMPILER_LIBS_URL "https://github.com/openvinotoolkit/npu_compiler/releases/download/npu_ud_2025_38_rc4/w_vpux_compiler_l0_win-7_4_3-Release_dyntbb_postcommit_cid_a826bd92b5e02af092e4d706a762252b1845f777_251010_2218.zip")
        set(PLUGIN_COMPILER_LIBS_ZIP "${PLUGIN_COMPILER_LIBS_DIR}/w_vpux_compiler_l0_win-7_4_3-Release_dyntbb_postcommit_cid_a826bd92b5e02af092e4d706a762252b1845f777_251010_2218.zip")
        set(PLUGIN_COMPILER_LIBS_DIR_UNZIPPED "${PLUGIN_COMPILER_LIBS_DIR}/cid_a826bd92b5e02af092e4d706a762252b1845f777_251010_2218")

        download_and_extract("${PLUGIN_COMPILER_LIBS_URL}" "${PLUGIN_COMPILER_LIBS_ZIP}" "${PLUGIN_COMPILER_LIBS_DIR_UNZIPPED}" "MODIFY")
        set(PLUGIN_COMPILER_LIB_PATH "${PLUGIN_COMPILER_LIBS_DIR_UNZIPPED}/cid/lib")

        configure_file(
            ${PLUGIN_COMPILER_LIB_PATH}/npu_driver_compiler.dll
            ${PLUGIN_COMPILER_LIB_PATH}/openvino_intel_npu_compiler.dll
            COPYONLY
        )
        set(PLUGIN_COMPILER_LIB "${PLUGIN_COMPILER_LIB_PATH}/openvino_intel_npu_compiler.dll")
        file(COPY "${PLUGIN_COMPILER_LIB}"
            DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}")
        message(STATUS "Not Copying prebuilt Plugin compiler libraries openvino_intel_npu_compiler.dll to ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} for windows")
    else()
        # Check if the operating system is Linux and not macOS
        if(UNIX AND NOT APPLE)
            # Get the OS name and version
            execute_process(COMMAND lsb_release -is OUTPUT_VARIABLE OS_NAME OUTPUT_STRIP_TRAILING_WHITESPACE)
            execute_process(COMMAND lsb_release -rs OUTPUT_VARIABLE OS_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

            if(OS_NAME STREQUAL "Ubuntu")
                if(OS_VERSION STREQUAL "22.04")
                    # Ubuntu 22.04-specific settings or actions
                    set(PLUGIN_COMPILER_LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/temp/compiler_libs/ubuntu22.04")
                    set(PLUGIN_COMPILER_LIBS_URL "https://github.com/openvinotoolkit/npu_compiler/releases/download/npu_ud_2025_44_rc1/l_vpux_compiler_l0_linux_ubuntu_22_04-7_4_3-Release_dyntbb_postcommit_cid_a934b15d7494c4961afd51cf6c896b15d1fabd8c_251119_2122.tar.gz")
                    set(PLUGIN_COMPILER_LIBS_TAR "${PLUGIN_COMPILER_LIBS_DIR}/l_vpux_compiler_l0_linux_ubuntu_22_04-7_4_3-Release_dyntbb_postcommit_cid_a934b15d7494c4961afd51cf6c896b15d1fabd8c_251119_2122.tar.gz")
                    set(PLUGIN_COMPILER_LIBS_DIR_EXTRACTED "${PLUGIN_COMPILER_LIBS_DIR}/7_4_3-Release_dyntbb_postcommit_cid_a934b15d7494c4961afd51cf6c896b15d1fabd8c_251119_212")

                    download_and_extract("${PLUGIN_COMPILER_LIBS_URL}" "${PLUGIN_COMPILER_LIBS_TAR}" "${PLUGIN_COMPILER_LIBS_DIR_EXTRACTED}" "NONE")
                    set(PLUGIN_COMPILER_LIB_PATH "${PLUGIN_COMPILER_LIBS_DIR_EXTRACTED}/cid/lib/")

                    configure_file(
                        ${PLUGIN_COMPILER_LIB_PATH}/libnpu_driver_compiler.so
                        ${PLUGIN_COMPILER_LIB_PATH}/libopenvino_intel_npu_compiler.so
                        COPYONLY
                    )
                    set(PLUGIN_COMPILER_LIB "${PLUGIN_COMPILER_LIB_PATH}/libopenvino_intel_npu_compiler.so")
                    file(COPY "${PLUGIN_COMPILER_LIB}" DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
                    message(STATUS "Not Copying prebuilt Plugin compiler libraries libopenvino_intel_npu_compiler.so to ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} for Ubuntu 22.04")
                elseif(OS_VERSION STREQUAL "24.04")
                    message(STATUS "This is Ubuntu 24.04")
                    set(PLUGIN_COMPILER_LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/temp/compiler_libs/ubuntu24.04")
                    set(PLUGIN_COMPILER_LIBS_URL "https://github.com/openvinotoolkit/npu_compiler/releases/download/npu_ud_2025_44_rc1/l_vpux_compiler_l0_linux_ubuntu_24_04-7_4_3-Release_dyntbb_postcommit_cid_a934b15d7494c4961afd51cf6c896b15d1fabd8c_251119_2204.tar.gz")
                    set(PLUGIN_COMPILER_LIBS_TAR "${PLUGIN_COMPILER_LIBS_DIR}/l_vpux_compiler_l0_linux_ubuntu_24_04-7_4_3-Release_dyntbb_postcommit_cid_a934b15d7494c4961afd51cf6c896b15d1fabd8c_251119_2204.tar.gz")
                    set(PLUGIN_COMPILER_LIBS_DIR_EXTRACTED "${PLUGIN_COMPILER_LIBS_DIR}/7_4_3-Release_dyntbb_postcommit_cid_a934b15d7494c4961afd51cf6c896b15d1fabd8c_251119_2204")

                    download_and_extract("${PLUGIN_COMPILER_LIBS_URL}" "${PLUGIN_COMPILER_LIBS_TAR}" "${PLUGIN_COMPILER_LIBS_DIR_EXTRACTED}" "NONE")
                    set(PLUGIN_COMPILER_LIB_PATH "${PLUGIN_COMPILER_LIBS_DIR_EXTRACTED}/cid/lib/")
                    configure_file(
                        ${PLUGIN_COMPILER_LIB_PATH}/libnpu_driver_compiler.so
                        ${PLUGIN_COMPILER_LIB_PATH}/libopenvino_intel_npu_compiler.so
                        COPYONLY
                    )
                    set(PLUGIN_COMPILER_LIB "${PLUGIN_COMPILER_LIB_PATH}/libopenvino_intel_npu_compiler.so")
                    file(COPY "${PLUGIN_COMPILER_LIB}" DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
                    message(STATUS "Copying prebuilt Plugin compiler libraries libopenvino_intel_npu_compiler.so to ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} for Ubuntu 24.04")
                endif()
            else()
            message(STATUS "This is a different Linux distribution: ${OS_NAME}, skip downloading prebuilt Plugin compiler libraries. Can not use plugin compiler libraries!")
                # Other Linux-specific settings or actions
            endif()
        endif()
    endif()

    install(FILES ${PLUGIN_COMPILER_LIB}
        DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${NPU_INTERNAL_COMPONENT})
endif()
