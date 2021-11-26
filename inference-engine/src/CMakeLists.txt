# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(CMAKE_COMPILER_IS_GNUCXX)
    ie_add_compiler_flags(-Wall)
    ie_add_compiler_flags(-Wmissing-declarations)
endif()

add_subdirectory(transformations)

add_subdirectory(low_precision_transformations)

add_subdirectory(offline_transformations)

add_subdirectory(snippets)

if(ENABLE_MKL_DNN)
    add_subdirectory(mkldnn_plugin)
endif()

if(ENABLE_CLDNN)
    add_subdirectory(cldnn_engine)
endif()

if(ENABLE_VPU)
    add_subdirectory(vpu)
endif()

if(ENABLE_GNA)
    add_subdirectory(gna_plugin)
endif()

if(ENABLE_HETERO)
    add_subdirectory(hetero_plugin)
endif()

if(ENABLE_MULTI)
    add_subdirectory(multi_device)
endif()

add_subdirectory(inference_engine)

add_subdirectory(legacy_api)

add_subdirectory(readers)

add_subdirectory(preprocessing)

# add a custom target to build all Inference Engine Core libraries

add_custom_target(ie_libraries ALL
                  DEPENDS inference_engine_transformations inference_engine_legacy
                          inference_engine inference_engine_preproc
                          inference_engine_lp_transformations inference_engine_snippets)

if(ENABLE_IR_V7_READER)
    add_dependencies(ie_libraries inference_engine_ir_v7_reader)
endif()

if(NGRAPH_IR_FRONTEND_ENABLE)
    if(BUILD_SHARED_LIBS)
        add_dependencies(ie_libraries ir_ov_frontend)
    endif()
    # use this one once CVS-69781 is fixed
    # add_dependencies(inference_engine ir_ov_frontend)
endif()

if(NGRAPH_ONNX_FRONTEND_ENABLE)
    add_dependencies(inference_engine onnx_ov_frontend)
endif()

if(NGRAPH_PDPD_FRONTEND_ENABLE)
    add_dependencies(inference_engine paddlepaddle_ov_frontend)
endif()

if(NGRAPH_TF_FRONTEND_ENABLE)
    add_dependencies(inference_engine tensorflow_ov_frontend)
endif()
