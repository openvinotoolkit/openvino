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
                           "${OpenVINO_SOURCE_DIR}/ngraph/python/pybind11"
                           "${IE_MAIN_SOURCE_DIR}/thirdparty"
                           "${TEMP}"
                           # TODO fix and enable back:
                           "${OpenVINO_SOURCE_DIR}/inference-engine/scripts/dependencies.sh"
                           "${OpenVINO_SOURCE_DIR}/scripts/install_dependencies/install_NEO_OCL_driver.sh"
                           "${OpenVINO_SOURCE_DIR}/scripts/install_dependencies/install_openvino_dependencies.sh"
                           "${OpenVINO_SOURCE_DIR}/ngraph/python/tests/test_onnx/model_zoo_preprocess.sh"
                           )

#
# cpack
#

# install setupvars

ie_cpack_add_component(setupvars REQUIRED)

if(UNIX)
    install(PROGRAMS setupvars/setupvars.sh
            DESTINATION bin
            COMPONENT setupvars)
elseif(WIN32)
    install(PROGRAMS setupvars/setupvars.bat
            DESTINATION bin
            COMPONENT setupvars)
endif()

# install install_dependencies

if(UNIX)
    ie_cpack_add_component(install_dependencies REQUIRED)
    install(DIRECTORY install_dependencies/
            DESTINATION install_dependencies
            COMPONENT install_dependencies)
endif()

# install DeploymentManager

ie_cpack_add_component(deployment_manager REQUIRED)
install(DIRECTORY deployment_manager/
        DESTINATION deployment_tools/tools/deployment_manager
        COMPONENT deployment_manager
        USE_SOURCE_PERMISSIONS)

# install files for demo

ie_cpack_add_component(demo_scripts DEPENDS core)

if(UNIX)
    install(DIRECTORY demo/
            DESTINATION deployment_tools/demo
            COMPONENT demo_scripts
            USE_SOURCE_PERMISSIONS
            PATTERN *.bat EXCLUDE)
elseif(WIN32)
    install(DIRECTORY demo/
            DESTINATION deployment_tools/demo
            COMPONENT demo_scripts
            USE_SOURCE_PERMISSIONS
            PATTERN *.sh EXCLUDE)
endif()
