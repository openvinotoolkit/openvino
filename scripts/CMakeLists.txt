# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# Shellcheck
#

ie_shellcheck_process(DIRECTORY "${OpenVINO_SOURCE_DIR}"
                      SKIP "${OpenVINO_SOURCE_DIR}/bin"
                           "${OpenVINO_SOURCE_DIR}/build"
                           "${OpenVINO_SOURCE_DIR}/thirdparty"
                           "${OpenVINO_SOURCE_DIR}/src/plugins/intel_cpu/thirdparty"
                           "${OpenVINO_SOURCE_DIR}/src/plugins/intel_gpu/thirdparty"
                           "${OpenVINO_SOURCE_DIR}/src/bindings/python/thirdparty/pybind11"
                           "${OpenVINO_SOURCE_DIR}/tools/pot/thirdparty"
                           "${TEMP}"
                           # TODO fix and enable back:
                           "${OpenVINO_SOURCE_DIR}/scripts/install_dependencies/install_NEO_OCL_driver.sh"
                           "${OpenVINO_SOURCE_DIR}/scripts/install_dependencies/install_openvino_dependencies.sh"
                           "${OpenVINO_SOURCE_DIR}/src/bindings/python/tests/test_onnx/model_zoo_preprocess.sh"
                           "${OpenVINO_SOURCE_DIR}/src/bindings/python/tests_compatibility/test_onnx/model_zoo_preprocess.sh"
                           )

#
# cpack
#

# install setupvars

ie_cpack_add_component(setupvars REQUIRED)

if(UNIX)
    set(_setupvars_file setupvars/setupvars.sh)
elseif(WIN32)
    set(_setupvars_file setupvars/setupvars.bat)
    if (USE_BUILD_TYPE_SUBFOLDER AND CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT CMAKE_BUILD_TYPE STREQUAL "Release")
        # Patch primary configuration in setupvars.bat which is "Release" by default.
        # Note setupvars secondary configuration is always "Debug".
        message(STATUS "Patching content of ${_setupvars_file} for CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
        file(READ "${_setupvars_file}" _setupvars_content)
        string(REPLACE "Release" ${CMAKE_BUILD_TYPE} _setupvars_content "${_setupvars_content}")
        set(_setupvars_file "${CMAKE_BINARY_DIR}/${_setupvars_file}")
        message(STATUS "Writing patched content to ${_setupvars_file}")
        file(WRITE "${_setupvars_file}" "${_setupvars_content}")
    endif()
endif()
install(PROGRAMS "${_setupvars_file}"
        DESTINATION .
        COMPONENT setupvars)

# install install_dependencies

if(LINUX)
    ie_cpack_add_component(install_dependencies REQUIRED)
    install(DIRECTORY install_dependencies/
            DESTINATION install_dependencies
            COMPONENT install_dependencies
            USE_SOURCE_PERMISSIONS)
endif()
