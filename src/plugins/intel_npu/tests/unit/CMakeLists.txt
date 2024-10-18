# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "ov_npu_unit_tests")

set(MANDATORY_UNIT_TESTS_LIBS
        "openvino::commonTestUtils"
        "openvino::gmock"
        "openvino::gtest"
        "openvino::gtest_main"
        "openvino::runtime"
        "openvino::npu_common"
        "openvino::npu_al"
        "openvino::npu_logger_utils"
)

ov_add_test_target(
        NAME ${TARGET_NAME}
        ROOT ${CMAKE_CURRENT_SOURCE_DIR}
        ADDITIONAL_SOURCE_DIRS
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw/
        DEPENDENCIES
            openvino::runtime
        INCLUDES
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/npuw
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/npuw
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/utils/include
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/plugin/include
            ${OpenVINO_SOURCE_DIR}/src/plugins/intel_npu/src/al/include
        LINK_LIBRARIES
            ${MANDATORY_UNIT_TESTS_LIBS}
        LABELS
            NPUW
)

if(ENABLE_AVX2)
    ov_avx2_optimization_flags(avx2_flags)
    target_compile_options(${TARGET_NAME} PRIVATE "${avx2_flags}")
endif()

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION tests
        COMPONENT tests
        EXCLUDE_FROM_ALL
)
