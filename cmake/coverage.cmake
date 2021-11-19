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

ie_coverage_extract(INPUT "openvino" OUTPUT "runtime"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/runtiem/*")
ie_coverage_genhtml(INFO_FILE "runtime"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "inference_engine_legacy"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/common/legacy_api/*")
ie_coverage_genhtml(INFO_FILE "inference_engine_legacy"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "hetero_plugin"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/inference-engine/src/hetero_plugin/*")
ie_coverage_genhtml(INFO_FILE "hetero_plugin"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "multi_device"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/inference-engine/src/multi_device/*")
ie_coverage_genhtml(INFO_FILE "multi_device"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "preprocessing"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/common/preprocessing/*")
ie_coverage_genhtml(INFO_FILE "preprocessing"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "inference_engine_transformations"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/inference-engine/src/inference_engine_transformations/*")
ie_coverage_genhtml(INFO_FILE "inference_engine_transformations"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "ov_snippets"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/common/snippets/*")
ie_coverage_genhtml(INFO_FILE "ov_snippets"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "low_precision_transformations"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/inference-engine/src/low_precision_transformations/*")
ie_coverage_genhtml(INFO_FILE "low_precision_transformations"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "template_plugin"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/docs/template_plugin/*")
ie_coverage_genhtml(INFO_FILE "template_plugin"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

if(ENABLE_MKL_DNN)
    ie_coverage_extract(INPUT "openvino" OUTPUT "mkldnn_plugin"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/inference-engine/src/mkldnn_plugin/*")
    ie_coverage_genhtml(INFO_FILE "mkldnn_plugin"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_CLDNN)
    ie_coverage_extract(INPUT "openvino" OUTPUT "cldnn_engine"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/inference-engine/src/cldnn_engine/*")
    ie_coverage_genhtml(INFO_FILE "cldnn_engine"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_GNA)
    ie_coverage_extract(INPUT "openvino" OUTPUT "gna_plugin"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/inference-engine/src/gna_plugin/*")
    ie_coverage_genhtml(INFO_FILE "gna_plugin"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

ie_coverage_extract(INPUT "openvino" OUTPUT "core"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/core/*")
ie_coverage_genhtml(INFO_FILE "core"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

if(OV_ONNX_FRONTEND_ENABLE)
    ie_coverage_extract(INPUT "openvino" OUTPUT "onnx_ov_frontend"
        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/frontends/onnx/*"
        "${OV_COVERAGE_BASE_DIRECTORY}/src/frontends/onnx/*")
    ie_coverage_genhtml(INFO_FILE "onnx_ov_frontend"
        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()
