# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# Python tools
#

# install deployment_manager

ov_cpack_add_component(${OV_CPACK_COMP_DEPLOYMENT_MANAGER} HIDDEN)

install(DIRECTORY deployment_manager
        DESTINATION ${OV_CPACK_TOOLSDIR}
        COMPONENT ${OV_CPACK_COMP_DEPLOYMENT_MANAGER}
        ${OV_CPACK_COMP_DEPLOYMENT_MANAGER_EXCLUDE_ALL}
        USE_SOURCE_PERMISSIONS)

# MO

add_subdirectory(mo)

# POT

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/pot/openvino/tools/pot/version.txt.in"
               "${CMAKE_CURRENT_SOURCE_DIR}/pot/openvino/tools/pot/version.txt" @ONLY)


if(ENABLE_PYTHON)

# Benchmark Tool
    add_subdirectory(benchmark_tool)
  
# OpenVino Conversion Tool
    add_subdirectory(ovc)

endif()

# wheel openvino-dev

add_subdirectory(openvino_dev)
