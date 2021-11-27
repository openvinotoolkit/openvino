# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
add_definitions(-DPROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(frontends)
# Core tests depend on frontends
add_subdirectory(core/tests)

if(ENABLE_PYTHON)
    add_subdirectory(bindings/python)
endif()
