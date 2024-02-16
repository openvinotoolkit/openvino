# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# OpenVINO npm binaries, includes openvino:runtime, frontends, plugins, tbb
#
macro(ov_cpack_settings)
    # fill a list of components which are part of NPM
    set(cpack_components_all ${CPACK_COMPONENTS_ALL})

    unset(CPACK_COMPONENTS_ALL)
    message(STATUS "Choosing CPack components for NPM...")
    foreach(item IN LISTS cpack_components_all)
        message(STATUS "${item}")
        string(TOUPPER ${item} UPPER_COMP)
        # filter out some components, which are not needed to be wrapped to npm package
        if(NOT OV_CPACK_COMP_${UPPER_COMP}_EXCLUDE_ALL AND
            # python is not required for npm package
            NOT item MATCHES "^${OV_CPACK_COMP_PYTHON_OPENVINO_PACKAGE}_python.*" AND
            NOT item MATCHES "tokenizers" AND
            NOT item MATCHES "java_api" AND
            NOT item MATCHES "nvidia" AND
            NOT item MATCHES "npu")
             list(APPEND CPACK_COMPONENTS_ALL ${item})
        endif()
    endforeach()
    unset(cpack_components_all)
    list(REMOVE_DUPLICATES CPACK_COMPONENTS_ALL)

    # override generator
    set(CPACK_GENERATOR "TGZ")
endmacro()
