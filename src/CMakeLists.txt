# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_definitions(-DPROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_target(ov_runtime_libraries)

include(cmake/install_tbb.cmake)

# TODO: Fix Wall for core, frontends, tests and  c binding
add_subdirectory(core)
add_subdirectory(frontends)
# TODO: remove ngraph/ngraph.hpp usage
add_subdirectory(core/tests)
add_subdirectory(bindings)

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wall)
    ie_add_compiler_flags(-Wmissing-declarations)
endif()

add_subdirectory(plugins)
add_subdirectory(inference)
include(cmake/ov_runtime.cmake)
add_subdirectory(common)

if(ENABLE_TESTS)
    add_subdirectory(tests_deprecated)
    add_subdirectory(tests)
endif()
