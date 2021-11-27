# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wall)
    ie_add_compiler_flags(-Wmissing-declarations)
endif()
add_definitions(-DPROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(inference)
add_subdirectory(frontends)
# Core tests depend on frontends
add_subdirectory(core/tests)

add_subdirectory(bindings)

if(NGRAPH_ONNX_FRONTEND_ENABLE)
    add_dependencies(inference_engine onnx_ov_frontend)
endif()

if(NGRAPH_PDPD_FRONTEND_ENABLE)
    add_dependencies(inference_engine paddlepaddle_ov_frontend)
endif()

if(NGRAPH_TF_FRONTEND_ENABLE)
    add_dependencies(inference_engine tensorflow_ov_frontend)
endif()
