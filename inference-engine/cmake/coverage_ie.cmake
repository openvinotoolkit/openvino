# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(DLDT_COVERAGE_GCDA_DATA_DIRECTORY "${CMAKE_BINARY_DIR}/inference-engine/src")
set(DLDT_COVERAGE_BASE_DIRECTORY "${IE_MAIN_SOURCE_DIR}/src")

ie_coverage_clean(REPOSITORY "dldt"
                  DIRECTORY "${DLDT_COVERAGE_GCDA_DATA_DIRECTORY}")
ie_coverage_capture(INFO_FILE "dldt"
                    BASE_DIRECTORY "${DLDT_COVERAGE_BASE_DIRECTORY}"
                    DIRECTORY "${DLDT_COVERAGE_GCDA_DATA_DIRECTORY}")

# Generate reports

ie_coverage_extract(INPUT "dldt" OUTPUT "inference_engine"
                    PATTERNS "${DLDT_COVERAGE_BASE_DIRECTORY}/inference_engine/*"
                             "${DLDT_COVERAGE_BASE_DIRECTORY}/plugin_api/*")
ie_coverage_genhtml(INFO_FILE "inference_engine"
                    PREFIX "${DLDT_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "dldt" OUTPUT "inference_engine_ir_reader"
                    PATTERNS "${DLDT_COVERAGE_BASE_DIRECTORY}/readers/*")
ie_coverage_genhtml(INFO_FILE "inference_engine_ir_reader"
                    PREFIX "${DLDT_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "dldt" OUTPUT "inference_engine_legacy"
                    PATTERNS "${DLDT_COVERAGE_BASE_DIRECTORY}/legacy_api/*")
ie_coverage_genhtml(INFO_FILE "inference_engine_legacy"
                    PREFIX "${DLDT_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "dldt" OUTPUT "hetero_plugin"
                    PATTERNS "${DLDT_COVERAGE_BASE_DIRECTORY}/hetero_plugin/*")
ie_coverage_genhtml(INFO_FILE "hetero_plugin"
                    PREFIX "${DLDT_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "dldt" OUTPUT "multi_device"
                    PATTERNS "${DLDT_COVERAGE_BASE_DIRECTORY}/multi_device/*")
ie_coverage_genhtml(INFO_FILE "multi_device"
                    PREFIX "${DLDT_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "dldt" OUTPUT "preprocessing"
                    PATTERNS "${DLDT_COVERAGE_BASE_DIRECTORY}/preprocessing/*")
ie_coverage_genhtml(INFO_FILE "preprocessing"
                    PREFIX "${DLDT_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "dldt" OUTPUT "inference_engine_transformations"
                    PATTERNS "${DLDT_COVERAGE_BASE_DIRECTORY}/inference_engine_transformations/*")
ie_coverage_genhtml(INFO_FILE "inference_engine_transformations"
                    PREFIX "${DLDT_COVERAGE_BASE_DIRECTORY}")

ie_coverage_extract(INPUT "dldt" OUTPUT "low_precision_transformations"
                    PATTERNS "${DLDT_COVERAGE_BASE_DIRECTORY}/low_precision_transformations/*")
ie_coverage_genhtml(INFO_FILE "low_precision_transformations"
                    PREFIX "${DLDT_COVERAGE_BASE_DIRECTORY}")

if(ENABLE_MKL_DNN)
    ie_coverage_extract(INPUT "dldt" OUTPUT "mkldnn_plugin"
                        PATTERNS "${DLDT_COVERAGE_BASE_DIRECTORY}/mkldnn_plugin/*")
    ie_coverage_genhtml(INFO_FILE "mkldnn_plugin"
                        PREFIX "${DLDT_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_CLDNN)
    ie_coverage_extract(INPUT "dldt" OUTPUT "cldnn_engine"
                        PATTERNS "${DLDT_COVERAGE_BASE_DIRECTORY}/cldnn_engine/*")
    ie_coverage_genhtml(INFO_FILE "cldnn_engine"
                        PREFIX "${DLDT_COVERAGE_BASE_DIRECTORY}")
endif()

if(ENABLE_GNA)
    ie_coverage_extract(INPUT "dldt" OUTPUT "gna_plugin"
                        PATTERNS "${DLDT_COVERAGE_BASE_DIRECTORY}/gna_plugin/*")
    ie_coverage_genhtml(INFO_FILE "gna_plugin"
                        PREFIX "${DLDT_COVERAGE_BASE_DIRECTORY}")
endif()
