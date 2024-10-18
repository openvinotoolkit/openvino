# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Components

macro(ov_cpack_settings)
    # fill a list of components which are part of NSIS or other GUI installer
    set(cpack_components_all ${CPACK_COMPONENTS_ALL})
    unset(CPACK_COMPONENTS_ALL)
    foreach(item IN LISTS cpack_components_all)
        # filter out some components, which are not needed to be wrapped to Windows package
        if(# python wheels are not needed to be wrapped by NSIS installer
           NOT item STREQUAL OV_CPACK_COMP_PYTHON_WHEELS AND
           # It was decided not to distribute JAX as C++ component
           NOT item STREQUAL "jax")
            list(APPEND CPACK_COMPONENTS_ALL ${item})
        endif()
    endforeach()
    unset(cpack_components_all)

    # restore the components settings

    foreach(comp IN LISTS CPACK_COMPONENTS_ALL)
        cpack_add_component(${comp} ${_${comp}_cpack_component_args})
    endforeach()

    # override package file name
    set(CPACK_PACKAGE_FILE_NAME "w_openvino_toolkit_p_${OpenVINO_VERSION}.${OpenVINO_VERSION_BUILD}_offline")
endmacro()
