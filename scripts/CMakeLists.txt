# Copyright (C) 2018-2025 Intel Corporation
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
    "${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/thirdparty"
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
    set(_setupvars_files "${CMAKE_CURRENT_SOURCE_DIR}/setupvars/setupvars.sh")
elseif(WIN32)
    # setupvars.bat
    set(_setupvars_file "setupvars/setupvars.bat")
    set(_setupvars_files "${CMAKE_CURRENT_SOURCE_DIR}/${_setupvars_file}")

    # setupvars.ps1
    set(_setupvars_file_pwsh "setupvars/setupvars.ps1")
    list(APPEND _setupvars_files "${CMAKE_CURRENT_SOURCE_DIR}/${_setupvars_file_pwsh}")
endif()

install(PROGRAMS ${_setupvars_files}
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
