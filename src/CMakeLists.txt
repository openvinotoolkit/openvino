# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
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
add_subdirectory(common)


if(ENABLE_OV_IR_FRONTEND)
    if(BUILD_SHARED_LIBS)
        add_dependencies(ov_runtime_libraries ir_ov_frontend)
    endif()
    # use this one once CVS-69781 is fixed
    # add_dependencies(inference_engine ir_ov_frontend)
endif()
if(ENABLE_OV_ONNX_FRONTEND)
    add_dependencies(inference_engine onnx_ov_frontend)
endif()

if(ENABLE_OV_PDPD_FRONTEND)
    add_dependencies(inference_engine paddlepaddle_ov_frontend)
endif()

if(ENABLE_OV_TF_FRONTEND)
    add_dependencies(inference_engine tensorflow_ov_frontend)
endif()
