# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_COVERAGE_BASE_DIRECTORY "${OpenVINO_SOURCE_DIR}")

ov_coverage_clean(REPOSITORY "openvino"
                  DIRECTORY "${OV_COVERAGE_GCDA_DATA_DIRECTORY}")
ov_coverage_capture(INFO_FILE "openvino"
                    BASE_DIRECTORY "${OV_COVERAGE_BASE_DIRECTORY}"
                    DIRECTORY "${OV_COVERAGE_GCDA_DATA_DIRECTORY}"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/*.pb.cc" 
                             "${OV_COVERAGE_BASE_DIRECTORY}/*.pb.h" 
                             "${OV_COVERAGE_BASE_DIRECTORY}/*/tests/*" 
                             "${OV_COVERAGE_BASE_DIRECTORY}/*/tests_deprecated/*" 
                             "${OV_COVERAGE_BASE_DIRECTORY}/thirdparty/*") # Skip some pb files, tests and thirdparty
ov_coverage_genhtml(INFO_FILE "openvino"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

# Generate reports

ov_coverage_extract(INPUT "openvino" OUTPUT "inference"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/inference/*")
ov_coverage_genhtml(INFO_FILE "inference"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ov_coverage_extract(INPUT "openvino" OUTPUT "core"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/core/*")
ov_coverage_genhtml(INFO_FILE "core"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ov_coverage_extract(INPUT "openvino" OUTPUT "transformations"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/common/transformations/*")
ov_coverage_genhtml(INFO_FILE "transformations"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ov_coverage_extract(INPUT "openvino" OUTPUT "low_precision_transformations"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/common/low_precision_transformations/*")
ov_coverage_genhtml(INFO_FILE "low_precision_transformations"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ov_coverage_extract(INPUT "openvino" OUTPUT "legacy"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/common/legacy/*")
ov_coverage_genhtml(INFO_FILE "legacy"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

if(ENABLE_INTEL_CPU)
    ov_coverage_extract(INPUT "openvino" OUTPUT "hetero_plugin"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/plugins/hetero/*")
    ov_coverage_genhtml(INFO_FILE "hetero_plugin"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_AUTO OR ENABLE_MULTI)
    ov_coverage_extract(INPUT "openvino" OUTPUT "auto_plugin"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/plugins/auto/*")
    ov_coverage_genhtml(INFO_FILE "auto_plugin"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

ov_coverage_extract(INPUT "openvino" OUTPUT "preprocessing"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}src/common/preprocessing/*")
ov_coverage_genhtml(INFO_FILE "preprocessing"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ov_coverage_extract(INPUT "openvino" OUTPUT "snippets"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/common/snippets/*")
ov_coverage_genhtml(INFO_FILE "snippets"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

if(ENABLE_TEMPLATE)
    ov_coverage_extract(INPUT "openvino" OUTPUT "template_plugin"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/docs/template_plugin/*")
    ov_coverage_genhtml(INFO_FILE "template_plugin"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_INTEL_CPU)
    ov_coverage_extract(INPUT "openvino" OUTPUT "intel_cpu_plugin"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/plugins/intel_cpu/*")
    ov_coverage_genhtml(INFO_FILE "intel_cpu_plugin"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if (ENABLE_INTEL_GPU)
    ov_coverage_extract(INPUT "openvino" OUTPUT "intel_gpu_plugin"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/plugins/intel_gpu/*")
    ov_coverage_genhtml(INFO_FILE "intel_gpu_plugin"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_INTEL_GNA)
    ov_coverage_extract(INPUT "openvino" OUTPUT "intel_gna_plugin"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/plugins/intel_gna/*")
    ov_coverage_genhtml(INFO_FILE "intel_gna_plugin"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_OV_ONNX_FRONTEND)
    ov_coverage_extract(INPUT "openvino" OUTPUT "onnx"
        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/frontends/onnx/*"
        "${OV_COVERAGE_BASE_DIRECTORY}/src/frontends/onnx/*")
    ov_coverage_genhtml(INFO_FILE "onnx"
        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()
