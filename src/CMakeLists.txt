# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(ENABLE_LTO)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

set(OV_CORE_INCLUDE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
)

add_custom_target(ov_runtime_libraries ALL)

add_subdirectory(runtime)
add_subdirectory(common)
add_subdirectory(core)

add_subdirectory(frontends)

# Enable tests because core tests depend on frontends
add_subdirectory(core/test)

if(ENABLE_IR_V7_READER)
    add_dependencies(ov_runtime_libraries inference_engine_ir_v7_reader)
endif()

if(OV_IR_FRONTEND_ENABLE)
    if(BUILD_SHARED_LIBS)
        add_dependencies(ov_runtime_libraries ir_ov_frontend)
    endif()
    # use this one once CVS-69781 is fixed
    # add_dependencies(inference_engine ir_ov_frontend)
endif()

if(OV_ONNX_FRONTEND_ENABLE)
    add_dependencies(ov_runtime_libraries onnx_ov_frontend)
endif()

if(OV_PDPD_FRONTEND_ENABLE)
    add_dependencies(ov_runtime_libraries paddlepaddle_ov_frontend)
endif()

if(OV_TF_FRONTEND_ENABLE)
    add_dependencies(ov_runtime_libraries tensorflow_ov_frontend)
endif()

add_subdirectory(bindings/c)
if(ENABLE_PYTHON)
    add_subdirectory(bindings/python)
endif()
