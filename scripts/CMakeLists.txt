# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Shellcheck
#

set(shellcheck_skip_list
    "${OpenVINO_SOURCE_DIR}/bin"
    "${OpenVINO_SOURCE_DIR}/build"
    "${OpenVINO_SOURCE_DIR}/thirdparty"
    "${OpenVINO_SOURCE_DIR}/src/plugins/intel_cpu/thirdparty"
    "${OpenVINO_SOURCE_DIR}/src/plugins/intel_gpu/thirdparty"
    "${OpenVINO_SOURCE_DIR}/src/bindings/python/thirdparty/pybind11"
    "${TEMP}")

ov_shellcheck_process(DIRECTORY "${OpenVINO_SOURCE_DIR}"
                      SKIP ${shellcheck_skip_list})

#
# cpack
#

# install setupvars

ov_cpack_add_component(${OV_CPACK_COMP_SETUPVARS} HIDDEN)

if(UNIX)
    set(_setupvars_file_full_path "${CMAKE_CURRENT_SOURCE_DIR}/setupvars/setupvars.sh")
elseif(WIN32)
    set(_setupvars_file "setupvars/setupvars.bat")
    set(_setupvars_file_full_path "${CMAKE_CURRENT_SOURCE_DIR}/${_setupvars_file}")
    if(USE_BUILD_TYPE_SUBFOLDER AND CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE MATCHES "^(Debug|Release)$")
        # Patch primary configuration in setupvars.bat which is "Release" by default.
        # Note setupvars secondary configuration is always "Debug".
        message(STATUS "Patching content of ${_setupvars_file_full_path} for CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
        file(READ "${_setupvars_file_full_path}" _setupvars_content)
        string(REPLACE "Release" ${CMAKE_BUILD_TYPE} _setupvars_content "${_setupvars_content}")
        set(_setupvars_file_full_path "${OpenVINO_BINARY_DIR}/${_setupvars_file}")
        message(STATUS "Writing patched content to ${_setupvars_file_full_path}")
        file(WRITE "${_setupvars_file_full_path}" "${_setupvars_content}")
    endif()
endif()

install(PROGRAMS "${_setupvars_file_full_path}"
        DESTINATION .
        COMPONENT ${OV_CPACK_COMP_SETUPVARS}
        ${OV_CPACK_COMP_SETUPVARS_EXCLUDE_ALL})

# install install_dependencies

if(LINUX)
    ov_cpack_add_component(${OV_CPACK_COMP_INSTALL_DEPENDENCIES} HIDDEN)

    install(PROGRAMS "${CMAKE_CURRENT_SOURCE_DIR}/install_dependencies/install_openvino_dependencies.sh"
            DESTINATION install_dependencies/
            COMPONENT ${OV_CPACK_COMP_INSTALL_DEPENDENCIES}
            ${OV_CPACK_COMP_INSTALL_DEPENDENCIES_EXCLUDE_ALL})
endif()
