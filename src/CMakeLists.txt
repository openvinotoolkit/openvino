# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_definitions(-DIN_OV_COMPONENT)

if(CMAKE_COMPILER_IS_GNUCXX OR OV_COMPILER_IS_CLANG)
    ov_add_compiler_flags(-Wmissing-declarations)
endif()

if(OV_COMPILER_IS_INTEL_LLVM)
    ov_disable_deprecated_warnings()
endif()

include(cmake/install_tbb.cmake)

# CC library should be registered before other cc targets
add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(frontends)

add_subdirectory(plugins)
add_subdirectory(inference)
include(cmake/openvino.cmake)
add_subdirectory(bindings)

if(ENABLE_TESTS)
    add_subdirectory(core/tests)
    add_subdirectory(tests)
endif()
