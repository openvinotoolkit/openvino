# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(vpu)

add_subdirectory(compile_tool)

# install

if(ENABLE_PYTHON)
    ie_cpack_add_component(python_tools DEPENDS core)

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_tool
            DESTINATION deployment_tools/tools
            COMPONENT python_tools)

    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cross_check_tool
            DESTINATION deployment_tools/tools
            COMPONENT python_tools)
endif()
