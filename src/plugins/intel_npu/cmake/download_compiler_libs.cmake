# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_VCL_FOR_COMPILER)
    if(ENABLE_SYSTEM_NPU_VCL_COMPILER)
        message(STATUS "Using system NPU VCL compiler libraries, skip download")
    else()
        message(STATUS "Downloading prebuilt NPU VCL compiler libraries")
        if(WIN32)
            set(VCL_COMPILER_LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/temp/vcl_compiler_lib/win")
            file(MAKE_DIRECTORY "${VCL_COMPILER_LIBS_DIR}")

            set(VCL_COMPILER_LIBS_URL "https://downloadmirror.intel.com/854488/npu_win_32.0.100.4023.zip")
            set(VCL_COMPILER_LIBS_ZIP "${VCL_COMPILER_LIBS_DIR}/npu_win_32.0.100.4023.zip")
            set(VCL_COMPILER_LIBS_DIR_UNZIPPED "${VCL_COMPILER_LIBS_DIR}/prebuilt_VCL_libs_from_32.0.100.4023")

            # Check if the prebuilt VCL compiler libraries not exist
            if(NOT EXISTS "${VCL_COMPILER_LIBS_DIR_UNZIPPED}")
                # Download the prebuilt VCL compiler libraries, if failure, show error message
                # and exit
                message(STATUS "Downloading prebuilt VCL compiler libraries from ${VCL_COMPILER_LIBS_URL}")
                file(DOWNLOAD "${VCL_COMPILER_LIBS_URL}" "${VCL_COMPILER_LIBS_ZIP}"
                    TIMEOUT 3600
                    LOG log_output
                    STATUS download_status
                    SHOW_PROGRESS)
                list(GET download_status 0 download_result)
                if(NOT download_result EQUAL 0)
                    message(FATAL_ERROR "Download failed!\nStatus: ${download_status}\nLog: ${log_output}")
                else()
                    message(STATUS "Download completed: ${VCL_COMPILER_LIBS_ZIP}")
                endif()

                message(STATUS "Unzipping prebuilt VCL compiler libraries to ${VCL_COMPILER_LIBS_DIR_UNZIPPED}")
		file(ARCHIVE_EXTRACT INPUT "${VCL_COMPILER_LIBS_ZIP}" DESTINATION "${VCL_COMPILER_LIBS_DIR_UNZIPPED}")
                file(REMOVE "${VCL_COMPILER_LIBS_ZIP}")
            else()
                message(STATUS "Prebuilt VCL compiler libraries already exist, skip download")
            endif()

            file(COPY ${VCL_COMPILER_LIBS_DIR_UNZIPPED}/npu_win_32.0.100.4023/drivers/x64/npu_driver_compiler.dll
	      DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}
            )
            message(STATUS "Copying prebuilt VCL compiler libraries npu_driver_compiler.dll to ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
        else()
            set(VCL_COMPILER_LIBS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/temp/vcl_compiler_libs/ubuntu22")
            file(MAKE_DIRECTORY "${VCL_COMPILER_LIBS_DIR}")

            set(VCL_COMPILER_LIBS_URL "https://github.com/intel/linux-npu-driver/releases/download/v1.17.0/intel-driver-compiler-npu_1.17.0.20250508-14912879441_ubuntu22.04_amd64.deb")
            set(VCL_COMPILER_LIBS_DEB "${VCL_COMPILER_LIBS_DIR}/intel-driver-compiler-npu_1.17.0.20250508-14912879441_ubuntu22.04_amd64.deb")
            set(VCL_COMPILER_LIBS_DIR_EXTRACTED "${VCL_COMPILER_LIBS_DIR}/prebuilt_VCL_libs_from_1.17.0.20250508-14912879441_ubuntu22.04")

            # Check if the prebuilt VCL compiler libraries not exist
            if(NOT EXISTS "${VCL_COMPILER_LIBS_DIR_EXTRACTED}")
                # Download the prebuilt VCL compiler libraries, if failure, show error message
                # and exit
                message(STATUS "Downloading prebuilt VCL compiler libraries from ${VCL_COMPILER_LIBS_URL}")
                file(DOWNLOAD "${VCL_COMPILER_LIBS_URL}" "${VCL_COMPILER_LIBS_DEB}"
                    TIMEOUT 3600
                    LOG log_output
                    STATUS download_status
                    SHOW_PROGRESS)
                list(GET download_status 0 download_result)
                if(NOT download_result EQUAL 0)
                    message(FATAL_ERROR "Download failed!\nStatus: ${download_status}\nLog: ${log_output}")
                else()
                    message(STATUS "Download completed: ${VCL_COMPILER_LIBS_DEB}")
                endif()

                message(STATUS "Unzipping prebuilt VCL compiler libraries to ${VCL_COMPILER_LIBS_DIR_EXTRACTED}")
                execute_process(COMMAND dpkg-deb -x ${VCL_COMPILER_LIBS_DEB} ${VCL_COMPILER_LIBS_DIR_EXTRACTED})
                file(REMOVE "${VCL_COMPILER_LIBS_DEB}")
            else()
                message(STATUS "Prebuilt VCL compiler libraries already exist, skip download")
            endif()
            file(COPY ${VCL_COMPILER_LIBS_DIR_EXTRACTED}/usr/lib/x86_64-linux-gnu/libnpu_driver_compiler.so
                DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}
            )
            message(STATUS "Copying prebuilt VCL compiler libraries libnpu_driver_compiler.so to ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
        endif()
    endif()
endif()
