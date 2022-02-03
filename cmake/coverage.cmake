# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_COVERAGE_BASE_DIRECTORY "${OpenVINO_SOURCE_DIR}")

ie_coverage_clean(REPOSITORY "openvino"
                  DIRECTORY "${OV_COVERAGE_GCDA_DATA_DIRECTORY}")
ie_coverage_capture(INFO_FILE "openvino"
                    BASE_DIRECTORY "${OV_COVERAGE_BASE_DIRECTORY}"
                    DIRECTORY "${OV_COVERAGE_GCDA_DATA_DIRECTORY}")

# Generate reports

ie_coverage_extract(INPUT "openvino" OUTPUT "inference"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/inference/*")

ie_coverage_genhtml(INFO_FILE "inference"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "legacy"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/common/legacy/*")
ie_coverage_genhtml(INFO_FILE "legacy"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "hetero_plugin"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/plugins/hetero/*")
ie_coverage_genhtml(INFO_FILE "hetero_plugin"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "auto_plugin"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/plugins/auto/*")
ie_coverage_genhtml(INFO_FILE "auto_plugin"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "preprocessing"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}src/common/preprocessing/*")
ie_coverage_genhtml(INFO_FILE "preprocessing"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "transformations"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/common/transformations/*")
ie_coverage_genhtml(INFO_FILE "transformations"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "snippets"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/common/snippets/*")
ie_coverage_genhtml(INFO_FILE "snippets"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "low_precision_transformations"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/common/low_precision_transformations/*")
ie_coverage_genhtml(INFO_FILE "low_precision_transformations"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "openvino" OUTPUT "template_plugin"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/docs/template_plugin/*")
ie_coverage_genhtml(INFO_FILE "template_plugin"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

if(ENABLE_INTEL_CPU)
    ie_coverage_extract(INPUT "openvino" OUTPUT "intel_cpu_plugin"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/plugins/intel_cpu/*")
    ie_coverage_genhtml(INFO_FILE "intel_cpu_plugin"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if (ENABLE_INTEL_GPU)
    ie_coverage_extract(INPUT "openvino" OUTPUT "intel_gpu_plugin"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/plugins/intel_gpu/*")
    ie_coverage_genhtml(INFO_FILE "intel_gpu_plugin"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_INTEL_GNA)
    ie_coverage_extract(INPUT "openvino" OUTPUT "intel_gna_plugin"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/plugins/intel_gna/*")
    ie_coverage_genhtml(INFO_FILE "intel_gna_plugin"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

ie_coverage_extract(INPUT "openvino" OUTPUT "core"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/core/*")
ie_coverage_genhtml(INFO_FILE "core"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

if(ENABLE_OV_ONNX_FRONTEND)
    ie_coverage_extract(INPUT "openvino" OUTPUT "onnx"
        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/frontends/onnx/*"
        "${OV_COVERAGE_BASE_DIRECTORY}/src/frontends/onnx/*")
    ie_coverage_genhtml(INFO_FILE "onnx"
        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()
