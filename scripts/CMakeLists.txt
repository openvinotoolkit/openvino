# Copyright (C) 2018-2021 Intel Corporation
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
                           "${OpenVINO_SOURCE_DIR}/src/bindings/python/thirdparty/pybind11"
                           "${IE_MAIN_SOURCE_DIR}/thirdparty"
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
    install(PROGRAMS setupvars/setupvars.sh
            DESTINATION .
            COMPONENT setupvars)
elseif(WIN32)
    install(PROGRAMS setupvars/setupvars.bat
            DESTINATION .
            COMPONENT setupvars)
endif()

# install install_dependencies

if(UNIX)
    ie_cpack_add_component(install_dependencies REQUIRED)
    install(DIRECTORY install_dependencies/
            DESTINATION install_dependencies
            COMPONENT install_dependencies
            USE_SOURCE_PERMISSIONS)
endif()
