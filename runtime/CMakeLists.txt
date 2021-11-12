# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

set(OV_CORE_INCLUDE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
)
set(OV_CORE_TARGETS_FILE "${CMAKE_CURRENT_BINARY_DIR}/coreTargets.cmake")

add_subdirectory(core)

add_subdirectory(frontends)

# Enable tests because core tests depend on frontends
add_subdirectory(core/test)

if(ENABLE_PYTHON)
    add_subdirectory(bindings/python)
endif()
