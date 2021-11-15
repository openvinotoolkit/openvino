# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(itt)
add_subdirectory(conditional_compilation)
add_subdirectory(util)
add_subdirectory(core)

openvino_developer_export_targets(COMPONENT openvino_common TARGETS openvino::itt openvino::conditional_compilation)

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wall)
    ie_add_compiler_flags(-Wmissing-declarations)
endif()

add_subdirectory(snippets)
add_subdirectory(runtime)
add_subdirectory(legacy_api)
add_subdirectory(preprocessing)

# add a custom target to build all OpenVINO runtime libraries

add_custom_target(ov_runtime_libraries ALL
                  DEPENDS inference_engine_legacy
                          openvino::runtime openvino::preprocessing
                          openvino::snippets)

if(OV_IR_FRONTEND_ENABLE)
    if(BUILD_SHARED_LIBS)
        add_dependencies(ov_runtime_libraries ir_ov_frontend)
    endif()
    # use this one once CVS-69781 is fixed
    # add_dependencies(ov_runtime ir_ov_frontend)
endif()

if(OV_ONNX_FRONTEND_ENABLE)
    add_dependencies(ov_runtime onnx_ov_frontend)
endif()

if(OV_PDPD_FRONTEND_ENABLE)
    add_dependencies(ov_runtime paddlepaddle_ov_frontend)
endif()

if(OV_TF_FRONTEND_ENABLE)
    add_dependencies(ov_runtime tensorflow_ov_frontend)
endif()
