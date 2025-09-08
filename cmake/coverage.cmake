# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(OV_COVERAGE_BASE_DIRECTORY "${OpenVINO_SOURCE_DIR}")

ov_coverage_clean(REPOSITORY "openvino"
                  DIRECTORY "${OV_COVERAGE_GCDA_DATA_DIRECTORY}")
ov_coverage_capture(INFO_FILE "openvino"
                    BASE_DIRECTORY "${OV_COVERAGE_BASE_DIRECTORY}"
                    DIRECTORY "${OV_COVERAGE_GCDA_DATA_DIRECTORY}"
                    EXCLUDE_PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/*.pb.cc"
                                     "${OV_COVERAGE_BASE_DIRECTORY}/*.pb.h"
                                     "${OV_COVERAGE_BASE_DIRECTORY}/*/tests/*"
                                     "${OV_COVERAGE_BASE_DIRECTORY}/*/tests_deprecated/*"
                                     "${OV_COVERAGE_BASE_DIRECTORY}/thirdparty/*"
                                     "${OV_COVERAGE_BASE_DIRECTORY}/CMakeCXXCompilerId.cpp"
                                     "${OV_COVERAGE_BASE_DIRECTORY}/CMakeCCompilerId.c") # Skip some service files, tests and thirdparty
# Generate reports

# Common report
ov_coverage_genhtml(INFO_FILE "openvino"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

##################### Core Components #####################
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

ov_coverage_extract(INPUT "openvino" OUTPUT "preprocessing"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}src/common/preprocessing/src/*")
ov_coverage_genhtml(INFO_FILE "preprocessing"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ov_coverage_extract(INPUT "openvino" OUTPUT "snippets"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/common/snippets/*")
ov_coverage_genhtml(INFO_FILE "snippets"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")

ov_coverage_extract(INPUT "openvino" OUTPUT "frontend_common"
                    PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/frontends/common/*")
ov_coverage_genhtml(INFO_FILE "frontend_common"
                    PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
##################### Core Components #####################


##################### Plugins #####################
if(ENABLE_AUTO OR ENABLE_MULTI)
    ov_coverage_extract(INPUT "openvino" OUTPUT "auto_plugin"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/plugins/auto/*")
    ov_coverage_genhtml(INFO_FILE "auto_plugin"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_AUTO_BATCH)
    ov_coverage_extract(INPUT "openvino" OUTPUT "auto_batch_plugin"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/plugins/auto_batch/*")
    ov_coverage_genhtml(INFO_FILE "auto_batch_plugin"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_HETERO)
    ov_coverage_extract(INPUT "openvino" OUTPUT "hetero_plugin"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/plugins/hetero/*")
    ov_coverage_genhtml(INFO_FILE "hetero_plugin"
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

if(ENABLE_TEMPLATE)
    ov_coverage_extract(INPUT "openvino" OUTPUT "template_plugin"
                        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/plugins/template/*")
    ov_coverage_genhtml(INFO_FILE "template_plugin"
                        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()
##################### Plugins #####################

##################### Frontends #####################
if(ENABLE_OV_IR_FRONTEND)
    ov_coverage_extract(INPUT "openvino" OUTPUT "ir_frontend"
        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/frontends/ir/*")
    ov_coverage_genhtml(INFO_FILE "ir_frontend"
        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_OV_JAX_FRONTEND)
    ov_coverage_extract(INPUT "openvino" OUTPUT "jax_frontend"
        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/frontends/jax/*")
    ov_coverage_genhtml(INFO_FILE "jax_frontend"
        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_OV_ONNX_FRONTEND)
    ov_coverage_extract(INPUT "openvino" OUTPUT "onnx_frontend"
        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/frontends/onnx/*")
    ov_coverage_genhtml(INFO_FILE "onnx_frontend"
        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_OV_PADDLE_FRONTEND)
    ov_coverage_extract(INPUT "openvino" OUTPUT "paddle_frontend"
        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/frontends/paddle/*")
    ov_coverage_genhtml(INFO_FILE "paddle_frontend"
        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_OV_PYTORCH_FRONTEND)
    ov_coverage_extract(INPUT "openvino" OUTPUT "pytorch_frontend"
        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/frontends/pytorch/*")
    ov_coverage_genhtml(INFO_FILE "pytorch_frontend"
        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_OV_TF_FRONTEND)
    ov_coverage_extract(INPUT "openvino" OUTPUT "tf_frontend"
        PATTERNS "${OV_COVERAGE_BASE_DIRECTORY}/src/frontends/tensorflow/*")
    ov_coverage_genhtml(INFO_FILE "tf_frontend"
        PREFIX "${OV_COVERAGE_BASE_DIRECTORY}")
endif()
##################### Frontends #####################
