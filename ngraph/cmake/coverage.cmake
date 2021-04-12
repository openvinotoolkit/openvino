# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
set(NGRAPH_COVERAGE_BASE_DIRECTORY "${OpenVINO_MAIN_SOURCE_DIR}")

ie_coverage_clean(REPOSITORY "nGraph"
                  DIRECTORY "${OV_COVERAGE_GCDA_DATA_DIRECTORY}")
ie_coverage_capture(INFO_FILE "nGraph"
                    BASE_DIRECTORY "${NGRAPH_COVERAGE_BASE_DIRECTORY}"
                    DIRECTORY "${OV_COVERAGE_GCDA_DATA_DIRECTORY}")

# Generate reports

ie_coverage_extract(INPUT "nGraph" OUTPUT "ngraph"
                    PATTERNS "${NGRAPH_COVERAGE_BASE_DIRECTORY}/ngraph/core/*")
ie_coverage_genhtml(INFO_FILE "ngraph"
                    PREFIX "${NGRAPH_COVERAGE_BASE_DIRECTORY}")
if (NGRAPH_ONNX_IMPORT_ENABLE)
    ie_coverage_extract(INPUT "nGraph" OUTPUT "onnx_importer"
        PATTERNS "${NGRAPH_COVERAGE_BASE_DIRECTORY}/ngraph/frontend/onnx_common*"
        "${NGRAPH_COVERAGE_BASE_DIRECTORY}/ngraph/frontend/onnx_import*")
    ie_coverage_genhtml(INFO_FILE "onnx_importer"
        PREFIX "${NGRAPH_COVERAGE_BASE_DIRECTORY}")
endif()
if (NGRAPH_ONNX_EDITOR_ENABLE)
    ie_coverage_extract(INPUT "nGraph" OUTPUT "onnx_editor"
        PATTERNS 
        "${NGRAPH_COVERAGE_BASE_DIRECTORY}/ngraph/frontend/onnx_editor*")
    ie_coverage_genhtml(INFO_FILE "onnx_editor"
        PREFIX "${NGRAPH_COVERAGE_BASE_DIRECTORY}")
endif()
