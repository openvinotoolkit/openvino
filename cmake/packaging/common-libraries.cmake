# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

#
# OpenVINO Core components including frontends, plugins, etc
#
macro(ov_cpack_settings)
    # fill a list of components which are part of conda
    set(cpack_components_all ${CPACK_COMPONENTS_ALL})
    unset(CPACK_COMPONENTS_ALL)
    foreach(item IN LISTS cpack_components_all)
        # filter out some components, which are not needed to be wrapped to conda-forge | brew
        if(# python is not a part of conda | brew
           NOT item MATCHES "^${OV_CPACK_COMP_PYTHON_OPENVINO}_python.*" AND
           # python wheels are not needed to be wrapped by conda | brew packages
           NOT item STREQUAL OV_CPACK_COMP_PYTHON_WHEELS AND
           # skip C / C++ / Python samples
           NOT item STREQUAL OV_CPACK_COMP_CPP_SAMPLES AND
           NOT item STREQUAL OV_CPACK_COMP_C_SAMPLES AND
           NOT item STREQUAL OV_CPACK_COMP_PYTHON_SAMPLES AND
           # even for case of system TBB we have installation rules for wheels packages
           # so, need to skip this explicitly since they are installed in `host` section
           NOT item MATCHES "^tbb(_dev)?$" AND
           # the same for pugixml
           NOT item STREQUAL "pugixml" AND
           # we have `license_file` field in conda meta.yml
           NOT item STREQUAL OV_CPACK_COMP_LICENSING AND
           # compile_tool is not needed
           NOT item STREQUAL OV_CPACK_COMP_CORE_TOOLS AND
           # not appropriate components
           NOT item STREQUAL OV_CPACK_COMP_DEPLOYMENT_MANAGER AND
           NOT item STREQUAL OV_CPACK_COMP_INSTALL_DEPENDENCIES AND
           NOT item STREQUAL OV_CPACK_COMP_SETUPVARS)
            list(APPEND CPACK_COMPONENTS_ALL ${item})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES CPACK_COMPONENTS_ALL)

    # override generator
    set(CPACK_GENERATOR "TGZ")
endmacro()
