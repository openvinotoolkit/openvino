# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_COVERAGE_BASE_DIRECTORY "${OpenVINO_SOURCE_DIR}")

ie_coverage_clean(REPOSITORY "openvino"
                  DIRECTORY "${OV_COVERAGE_GCDA_DATA_DIRECTORY}")
ie_coverage_capture(INFO_FILE "openvino"
                    BASE_DIRECTORY "${OV_COVERAGE_BASE_DIRECTORY}"
                    DIRECTORY "${OV_COVERAGE_GCDA_DATA_DIRECTORY}")

# Generate reports

ie_coverage_extract(INPUT "openvino" OUTPUT "inference_engine"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/inference_engine/*"
                             "${OV_COVERAGE_BASE_DIRECTORY}/plugin_api/*")
ie_coverage_genhtml(INFO_FILE "inference_engine"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "inference_engine_ir_v10_reader"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/readers/ir_reader/*"
                             "${OV_COVERAGE_BASE_DIRECTORY}/readers/reader_api/*")
ie_coverage_genhtml(INFO_FILE "inference_engine_ir_v10_reader"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "inference_engine_legacy"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/legacy_api/*")
ie_coverage_genhtml(INFO_FILE "inference_engine_legacy"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "hetero_plugin"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/hetero_plugin/*")
ie_coverage_genhtml(INFO_FILE "hetero_plugin"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "multi_device"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/multi_device/*")
ie_coverage_genhtml(INFO_FILE "multi_device"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "preprocessing"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/preprocessing/*")
ie_coverage_genhtml(INFO_FILE "preprocessing"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "inference_engine_transformations"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/inference_engine_transformations/*")
ie_coverage_genhtml(INFO_FILE "inference_engine_transformations"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "inference_engine_snippets"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/snippets/*")
ie_coverage_genhtml(INFO_FILE "inference_engine_snippets"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "low_precision_transformations"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/low_precision_transformations/*")
ie_coverage_genhtml(INFO_FILE "low_precision_transformations"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "template_plugin"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/template_plugin/*")
ie_coverage_genhtml(INFO_FILE "template_plugin"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

if(ENABLE_MKL_DNN)
    ie_coverage_extract(INPUT "openvino" OUTPUT "mkldnn_plugin"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/mkldnn_plugin/*")
    ie_coverage_genhtml(INFO_FILE "mkldnn_plugin"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_CLDNN)
    ie_coverage_extract(INPUT "openvino" OUTPUT "cldnn_engine"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/cldnn_engine/*")
    ie_coverage_genhtml(INFO_FILE "cldnn_engine"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_GNA)
    ie_coverage_extract(INPUT "openvino" OUTPUT "gna_plugin"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/gna_plugin/*")
    ie_coverage_genhtml(INFO_FILE "gna_plugin"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

ie_coverage_extract(INPUT "openvino" OUTPUT "ngraph"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/ngraph/core/*")
ie_coverage_genhtml(INFO_FILE "ngraph"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

if(NGRAPH_ONNX_IMPORT_ENABLE)
    ie_coverage_extract(INPUT "openvino" OUTPUT "onnx_importer"
        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/ngraph/frontend/onnx_common*"
                 "${OV_COVERAGE_BASE_DIRECTORY}/ngraph/frontend/onnx_editor*"
        "${OV_COVERAGE_BASE_DIRECTORY}/ngraph/frontend/onnx_import*")
    ie_coverage_genhtml(INFO_FILE "onnx_importer"
        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()
