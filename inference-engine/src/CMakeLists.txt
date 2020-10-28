# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

add_subdirectory(preprocessing)

add_subdirectory(readers)

add_subdirectory(legacy_api)

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

add_subdirectory(hetero_plugin)

add_subdirectory(multi_device)

add_subdirectory(transformations)

add_subdirectory(inference_engine)

add_subdirectory(low_precision_transformations_legacy)

add_subdirectory(low_precision_transformations)

# add a custom target to build all Inference Engine Core libraries

add_custom_target(ie_libraries ALL
                  DEPENDS inference_engine_transformations inference_engine_legacy
                          inference_engine inference_engine_preproc
                          inference_engine_ir_v7_reader inference_engine_ir_reader
                          inference_engine_lp_transformations_legacy
                          inference_engine_lp_transformations)

if(NGRAPH_ONNX_IMPORT_ENABLE)
    add_dependencies(ie_libraries inference_engine_onnx_reader)
endif()
