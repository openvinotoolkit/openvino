# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_definitions(-DPROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

include(cmake/install_tbb.cmake)

# CC library should be registered before other cc targets
add_subdirectory(common)
# TODO: Fix Wall for core, frontends, tests and  c binding
add_subdirectory(core)
add_subdirectory(frontends)
# TODO: remove ngraph/ngraph.hpp usage
if(ENABLE_TESTS)
    add_subdirectory(core/tests)
endif()
add_subdirectory(bindings)

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wall)
    ie_add_compiler_flags(-Wmissing-declarations)
endif()

add_subdirectory(plugins)
add_subdirectory(inference)
include(cmake/openvino.cmake)
# preprocessing has dependency on `openvino` for static build
add_subdirectory(common/preprocessing)

if(ENABLE_TESTS)
    add_subdirectory(tests_deprecated)
    add_subdirectory(tests)
endif()
