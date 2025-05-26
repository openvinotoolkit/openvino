# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Function to download and extract files
function(download_and_extract url dest_dir zip_file extracted_dir modify_proxy)
    # Check if the prebuilt VCL compiler libraries not exist
    if(NOT EXISTS "${extracted_dir}")
        if(modify_proxy STREQUAL "MODIFY")
            # Update proxy to enable download for windows url
	    set(original_NO_PROXY $ENV{NO_PROXY})
            set(original_no_proxy $ENV{no_proxy})
            set(ENV{NO_PROXY} "")
            set(ENV{no_proxy} "")
        endif()

        # Download the prebuilt VCL compiler libraries, if failure, show error message and exit
        message(STATUS "Downloading prebuilt VCL compiler libraries from ${url}")
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

        message(STATUS "Unzipping prebuilt VCL compiler libraries to ${extracted_dir}")
        # Determine extraction method based on file extension
        if("${zip_file}" MATCHES "\\.zip$")
            file(ARCHIVE_EXTRACT INPUT "${zip_file}" DESTINATION "${extracted_dir}")
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
        message(STATUS "Prebuilt VCL compiler libraries already exist, skip download")
    endif()
endfunction()

if(ENABLE_VCL_FOR_COMPILER)
    if(ENABLE_SYSTEM_NPU_VCL_COMPILER)
        message(STATUS "Using system NPU VCL compiler libraries, skip download")
    else()
        message(STATUS "Downloading prebuilt NPU VCL compiler libraries")
        if(WIN32)
            set(VCL_COMPILER_LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/temp/vcl_compiler_lib/win")
            set(VCL_COMPILER_LIBS_URL "https://downloadmirror.intel.com/854488/npu_win_32.0.100.4023.zip")
            set(VCL_COMPILER_LIBS_ZIP "${VCL_COMPILER_LIBS_DIR}/npu_win_32.0.100.4023.zip")
            set(VCL_COMPILER_LIBS_DIR_UNZIPPED "${VCL_COMPILER_LIBS_DIR}/npu_win_32.0.100.4023")

            download_and_extract("${VCL_COMPILER_LIBS_URL}" "${VCL_COMPILER_LIBS_DIR}" "${VCL_COMPILER_LIBS_ZIP}" "${VCL_COMPILER_LIBS_DIR_UNZIPPED}" "MODIFY")
            set(VCL_COMPILER_LIB_PATH "${VCL_COMPILER_LIBS_DIR_UNZIPPED}/npu_win_32.0.100.4023/drivers/x64/")


            configure_file(
                ${VCL_COMPILER_LIB_PATH}/npu_driver_compiler.dll
                ${VCL_COMPILER_LIB_PATH}/npu_vcl_compiler.dll
                COPYONLY
            )
            set(VCL_COMPILER_LIB "${VCL_COMPILER_LIB_PATH}/npu_vcl_compiler.dll")
            file(COPY "${VCL_COMPILER_LIB}"
                DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}")
            message(STATUS "Copying prebuilt VCL compiler libraries npu_vcl_compiler.dll to ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} for windows")
        else()
            # Check if the operating system is Linux and not macOS
            if(UNIX AND NOT APPLE)
                # Get the OS name and version
                execute_process(COMMAND lsb_release -is OUTPUT_VARIABLE OS_NAME OUTPUT_STRIP_TRAILING_WHITESPACE)
                execute_process(COMMAND lsb_release -rs OUTPUT_VARIABLE OS_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE)

                if(OS_NAME STREQUAL "Ubuntu")
                    if(OS_VERSION STREQUAL "22.04")
                        # Ubuntu 22.04-specific settings or actions
                        set(VCL_COMPILER_LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/temp/vcl_compiler_libs/ubuntu22.04")
                        set(VCL_COMPILER_LIBS_URL "https://github.com/intel/linux-npu-driver/releases/download/v1.19.0/intel-driver-compiler-npu_1.19.0.20250707-16111289554_ubuntu22.04_amd64.deb")
                        set(VCL_COMPILER_LIBS_DEB "${VCL_COMPILER_LIBS_DIR}/intel-driver-compiler-npu_1.19.0.20250707-16111289554_ubuntu22.04_amd64.deb")
                        set(VCL_COMPILER_LIBS_DIR_EXTRACTED "${VCL_COMPILER_LIBS_DIR}/prebuilt_VCL_libs_from_1.19.0.20250707-16111289554_ubuntu22.04")

                        download_and_extract("${VCL_COMPILER_LIBS_URL}" "${VCL_COMPILER_LIBS_DIR}" "${VCL_COMPILER_LIBS_DEB}" "${VCL_COMPILER_LIBS_DIR_EXTRACTED}" "NONE")

                        set(VCL_COMPILER_LIB_PATH "${VCL_COMPILER_LIBS_DIR_EXTRACTED}/usr/lib/x86_64-linux-gnu")
                        configure_file(
                            ${VCL_COMPILER_LIB_PATH}/libnpu_driver_compiler.so
                            ${VCL_COMPILER_LIB_PATH}/libnpu_vcl_compiler.so
                            COPYONLY
                        )
                        set(VCL_COMPILER_LIB "${VCL_COMPILER_LIB_PATH}/libnpu_vcl_compiler.so")
                        file(COPY "${VCL_COMPILER_LIB}"
                            DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
                        message(STATUS "Copying prebuilt VCL compiler libraries libnpu_vcl_compiler.so to ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} for Ubuntu 22.04")
                    elseif(OS_VERSION STREQUAL "24.04")
                        message(STATUS "This is Ubuntu 24.04")
                        # Ubuntu 24.04-specific settings or actions
                        set(VCL_COMPILER_LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/temp/vcl_compiler_libs/ubuntu24.04")
                        set(VCL_COMPILER_LIBS_URL "https://github.com/intel/linux-npu-driver/releases/download/v1.19.0/intel-driver-compiler-npu_1.19.0.20250707-16111289554_ubuntu24.04_amd64.deb")
                        set(VCL_COMPILER_LIBS_DEB "${VCL_COMPILER_LIBS_DIR}/intel-driver-compiler-npu_1.19.0.20250707-16111289554_ubuntu24.04_amd64.deb")
                        set(VCL_COMPILER_LIBS_DIR_EXTRACTED "${VCL_COMPILER_LIBS_DIR}/prebuilt_VCL_libs_from_1.19.0.20250707-16111289554_ubuntu24.04")

                        download_and_extract("${VCL_COMPILER_LIBS_URL}" "${VCL_COMPILER_LIBS_DIR}" "${VCL_COMPILER_LIBS_DEB}" "${VCL_COMPILER_LIBS_DIR_EXTRACTED}" "NONE")

                        set(VCL_COMPILER_LIB_PATH "${VCL_COMPILER_LIBS_DIR_EXTRACTED}/usr/lib/x86_64-linux-gnu")
                        configure_file(
                            ${VCL_COMPILER_LIB_PATH}/libnpu_driver_compiler.so
                            ${VCL_COMPILER_LIB_PATH}/libnpu_vcl_compiler.so
                            COPYONLY
                        )
                        set(VCL_COMPILER_LIB "${VCL_COMPILER_LIB_PATH}/libnpu_vcl_compiler.so")
                        file(COPY "${VCL_COMPILER_LIB}"
                            DESTINATION "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
                        message(STATUS "Copying prebuilt VCL compiler libraries libnpu_vcl_compiler.so to ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} for Ubuntu 24.04")
                    else()
                        message(STATUS "This is another version of Ubuntu: ${OS_VERSION}")
                        # Other Ubuntu-specific settings or actions
                    endif()
                else()
                    message(STATUS "This is a different Linux distribution: ${OS_NAME}, skip downloading prebuilt VCL compiler libraries")
                    # Other Linux-specific settings or actions
                endif()
            endif()
        endif()
    endif()

    install(FILES ${VCL_COMPILER_LIB}
        DESTINATION ${OV_CPACK_RUNTIMEDIR} COMPONENT ${NPU_INTERNAL_COMPONENT})
endif()
