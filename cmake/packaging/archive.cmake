# Copyright (C) 2018-2026 Intel Corporation
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
        string(TOUPPER ${item} UPPER_COMP)
        # filter out some components, which are not needed to be wrapped to conda-forge | brew | conan | vcpkg
        if(NOT OV_CPACK_COMP_${UPPER_COMP}_EXCLUDE_ALL AND
           # python_package is not needed in case of archives, because components like pyopenvino are used, as well as wheels
           NOT item MATCHES "^${OV_CPACK_COMP_PYTHON_OPENVINO_PACKAGE}_python.*" AND
           # It was decided not to distribute JAX as C++ component
           NOT item STREQUAL "jax")
            list(APPEND CPACK_COMPONENTS_ALL ${item})
        endif()
    endforeach()
    unset(cpack_components_all)
    list(REMOVE_DUPLICATES CPACK_COMPONENTS_ALL)

    # Multi-flavor archive packaging (opt-in via -DOV_CPACK_ARCHIVE_FLAVORS=ON).
    # Produces two archives from a single build: the full OpenVINO archive and a
    # reduced "winget" archive without samples and Python. The flavor is selected
    # at packaging time (cpack -D OV_CPACK_ARCHIVE_FLAVOR=full|winget) and applied
    # by the per-run project config file, which redefines CPACK_COMPONENTS_ALL.
    if(OV_CPACK_ARCHIVE_FLAVORS)
        # a single monolithic archive per flavor that still honors the component
        # list: COMPONENT_INSTALL must stay ON so setting CPACK_COMPONENTS_ALL
        # takes effect, while ALL_COMPONENTS_IN_ONE bundles them into one archive
        set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
        set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)
        set(CPACK_PROJECT_CONFIG_FILE "${OpenVINO_SOURCE_DIR}/cmake/packaging/archive_flavors.cmake")
    endif()
endmacro()
