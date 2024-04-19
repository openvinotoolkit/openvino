// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "onnx_utils.hpp"
#include "openvino/frontend/manager.hpp"

using namespace ov::frontend::onnx::tests;

TEST(ONNXFeConvertException, exception_if_node_unsupported) {
    OV_EXPECT_THROW(
        convert_model("unsupported_ops/add_unsupported.onnx"),
        ov::AssertFailure,
        testing::EndsWith("OpenVINO does not support the following ONNX operations: test_domain.UnsupportedAdd\n"));
}

TEST(ONNXFeConvertException, exception_if_more_nodes_unsupported) {
    OV_EXPECT_THROW(
        convert_model("unsupported_ops/two_unsupported_nodes.onnx"),
        ov::AssertFailure,
        testing::EndsWith("OpenVINO does not support the following ONNX operations: UnsupportedAdd, UnsupportedAbs\n"));
}

TEST(ONNXFeConvertException, exception_if_onnx_validation_exception) {
    OV_EXPECT_THROW(convert_model("instance_norm_bad_scale_type.onnx"),
                    ov::AssertFailure,
                    testing::EndsWith("Element types for data and scale input do not match (data element type: f32, "
                                      "scale element type: u16).\n"));
}

TEST(ONNXFeConvertException, exception_if_other_translation_exception) {
    OV_EXPECT_THROW(convert_model("depth_to_space_bad_mode.onnx"),
                    ov::AssertFailure,
                    testing::EndsWith("only 'DCR' and 'CRD' modes are supported\n"));
}

TEST(ONNXFeConvertException, exception_if_both_unsupported_and_other_translation_exception) {
    OV_EXPECT_THROW(
        convert_model("unsupported_ops/unsupported_add_and_incorrect_dts.onnx"),
        ov::AssertFailure,
        testing::HasSubstr("OpenVINO does not support the following ONNX operations: test_domain.UnsupportedAdd"));
    OV_EXPECT_THROW(convert_model("unsupported_ops/unsupported_add_and_incorrect_dts.onnx"),
                    ov::AssertFailure,
                    testing::EndsWith("only 'DCR' and 'CRD' modes are supported\n"));
}

TEST(ONNXFeConvertException, exception_if_both_unsupported_onnx_validation_exception_and_other_exception) {
    OV_EXPECT_THROW(
        convert_model("unsupported_ops/unsupported_add_incorrect_dts_and_inst_norm_bad_scale.onnx"),
        ov::AssertFailure,
        testing::HasSubstr("OpenVINO does not support the following ONNX operations: test_domain.UnsupportedAdd"));
    OV_EXPECT_THROW(convert_model("unsupported_ops/unsupported_add_incorrect_dts_and_inst_norm_bad_scale.onnx"),
                    ov::AssertFailure,
                    testing::HasSubstr("'stop' input is not a scalar"));
    OV_EXPECT_THROW(convert_model("unsupported_ops/unsupported_add_incorrect_dts_and_inst_norm_bad_scale.onnx"),
                    ov::AssertFailure,
                    testing::HasSubstr("only 'DCR' and 'CRD' modes are supported\n"));
}
