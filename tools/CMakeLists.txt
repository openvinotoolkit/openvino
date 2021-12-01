# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# C++ tools
#

add_subdirectory(compile_tool)

#
# Python tools
#

# install deployment_manager

ie_cpack_add_component(deployment_manager REQUIRED)

install(DIRECTORY deployment_manager
        DESTINATION tools
        COMPONENT deployment_manager
        USE_SOURCE_PERMISSIONS)

# build and install openvino-dev wheel

if(ENABLE_WHEEL)
    add_subdirectory(openvino_dev)
endif()
