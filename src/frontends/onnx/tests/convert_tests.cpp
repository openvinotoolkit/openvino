// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "openvino/frontend/manager.hpp"

namespace {
void convert(const std::string& path) {
    ov::frontend::FrontEndManager manager;
    const auto frontend = manager.load_by_framework("onnx");
    const auto in_model = frontend->load(path);
    frontend->convert(in_model);
}
}  // namespace

TEST(ONNXFeConvertException, exception_if_node_unsupported) {
    const auto path = ov::test::utils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) +
                                                                "unsupported_ops/add_unsupported.onnx");

    OV_EXPECT_THROW(
        convert(path),
        ov::AssertFailure,
        testing::EndsWith("OpenVINO does not support the following ONNX operations: test_domain.UnsupportedAdd\n"));
}

TEST(ONNXFeConvertException, exception_if_more_nodes_unsupported) {
    const auto path = ov::test::utils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) +
                                                                "unsupported_ops/two_unsupported_nodes.onnx");

    OV_EXPECT_THROW(
        convert(path),
        ov::AssertFailure,
        testing::EndsWith("OpenVINO does not support the following ONNX operations: UnsupportedAbs, UnsupportedAdd\n"));
}

TEST(ONNXFeConvertException, exception_if_onnx_validation_exception) {
    const auto path =
        ov::test::utils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) + "instance_norm_bad_scale_type.onnx");

    OV_EXPECT_THROW(convert(path),
                    ov::AssertFailure,
                    testing::EndsWith("Element types for data and scale input do not match (data element type: f32, "
                                      "scale element type: u16).\n"));
}

TEST(ONNXFeConvertException, exception_if_other_translation_exception) {
    const auto path =
        ov::test::utils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) + "depth_to_space_bad_mode.onnx");

    OV_EXPECT_THROW(convert(path), ov::AssertFailure, testing::EndsWith("only 'DCR' and 'CRD' modes are supported\n"));
}

TEST(ONNXFeConvertException, exception_if_both_unsupported_and_other_translation_exception) {
    const auto path = ov::test::utils::getModelFromTestModelZoo(
        std::string(ONNX_TEST_MODELS) + "unsupported_ops/unsupported_add_and_incorrect_dts.onnx");

    OV_EXPECT_THROW(
        convert(path),
        ov::AssertFailure,
        testing::HasSubstr("OpenVINO does not support the following ONNX operations: test_domain.UnsupportedAdd"));
    OV_EXPECT_THROW(convert(path), ov::AssertFailure, testing::EndsWith("only 'DCR' and 'CRD' modes are supported\n"));
}

TEST(ONNXFeConvertException, exception_if_both_unsupported_onnx_validation_exception_and_other_exception) {
    const auto path = ov::test::utils::getModelFromTestModelZoo(
        std::string(ONNX_TEST_MODELS) + "unsupported_ops/unsupported_add_incorrect_dts_and_inst_norm_bad_scale.onnx");

    OV_EXPECT_THROW(
        convert(path),
        ov::AssertFailure,
        testing::HasSubstr("OpenVINO does not support the following ONNX operations: test_domain.UnsupportedAdd"));
    OV_EXPECT_THROW(convert(path), ov::AssertFailure, testing::HasSubstr("'stop' input is not a scalar"));
    OV_EXPECT_THROW(convert(path), ov::AssertFailure, testing::EndsWith("only 'DCR' and 'CRD' modes are supported\n"));
}
