# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# C++ tools
#

if(ENABLE_COMPILE_TOOL)
    add_subdirectory(compile_tool)
endif()

if(ENABLE_SAMPLES)
    add_subdirectory(legacy/benchmark_app)
endif()

#
# Python tools
#

# install deployment_manager

ie_cpack_add_component(${OV_CPACK_COMP_DEPLOYMENT_MANAGER} HIDDEN)

install(DIRECTORY deployment_manager
        DESTINATION ${OV_CPACK_TOOLSDIR}
        COMPONENT ${OV_CPACK_COMP_DEPLOYMENT_MANAGER}
        USE_SOURCE_PERMISSIONS)

# MO

add_subdirectory(mo)

# POT

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/pot/openvino/tools/pot/version.txt.in"
               "${CMAKE_CURRENT_SOURCE_DIR}/pot/openvino/tools/pot/version.txt" @ONLY)

# wheel openvino-dev

add_subdirectory(openvino_dev)
