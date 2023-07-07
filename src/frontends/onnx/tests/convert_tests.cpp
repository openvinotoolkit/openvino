// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <file_utils.h>
#include <gtest/gtest.h>

#include <string>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include "openvino/frontend/manager.hpp"

namespace {
void convert(const std::string& path) {
    ov::frontend::FrontEndManager manager;
    const auto frontend = manager.load_by_framework("onnx");
    const auto in_model = frontend->load(path);
    frontend->convert(in_model);
}
bool has_suffix(std::string exception_message, std::string suffix) {
    return exception_message.size() >= suffix.size() &&
           exception_message.compare(exception_message.size() - suffix.size(), suffix.size(), suffix) == 0;
}
}  // namespace

TEST(ONNXFeConvertException, exception_if_node_unsupported) {
    const auto path = CommonTestUtils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) +
                                                                "unsupported_ops/add_unsupported.onnx");
    try {
        convert(path);
        FAIL() << "The exception should be thrown after conversion";
    } catch (const ov::AssertFailure& e) {
        EXPECT_TRUE(
            has_suffix(e.what(),
                       "OpenVINO does not support the following ONNX operations: test_domain.UnsupportedAdd\n"));
    } catch (...) {
        FAIL() << "Unexpected exception was thrown";
    }
}

TEST(ONNXFeConvertException, exception_if_more_nodes_unsupported) {
    const auto path = CommonTestUtils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) +
                                                                "unsupported_ops/two_unsupported_nodes.onnx");
    try {
        convert(path);
        FAIL() << "The exception should be thrown after conversion";
    } catch (const ov::AssertFailure& e) {
        EXPECT_TRUE(
            has_suffix(e.what(),
                       "OpenVINO does not support the following ONNX operations: UnsupportedAbs, UnsupportedAdd\n"));
    } catch (...) {
        FAIL() << "Unexpected exception was thrown";
    }
}

TEST(ONNXFeConvertException, exception_if_onnx_validation_exception) {
    const auto path =
        CommonTestUtils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) + "instance_norm_bad_scale_type.onnx");
    try {
        convert(path);
        FAIL() << "The exception should be thrown after conversion";
    } catch (const ov::AssertFailure& e) {
        EXPECT_TRUE(has_suffix(e.what(),
                               "Element types for data and scale input do not match (data element type: f32, scale "
                               "element type: u16).\n"));
    } catch (...) {
        FAIL() << "Unexpected exception was thrown";
    }
}

TEST(ONNXFeConvertException, exception_if_other_translation_exception) {
    const auto path =
        CommonTestUtils::getModelFromTestModelZoo(std::string(ONNX_TEST_MODELS) + "depth_to_space_bad_mode.onnx");
    try {
        convert(path);
        FAIL() << "The exception should be thrown after conversion";
    } catch (const ov::AssertFailure& e) {
        EXPECT_TRUE(has_suffix(e.what(), "only 'DCR' and 'CRD' modes are supported\n"));
    } catch (...) {
        FAIL() << "Unexpected exception was thrown";
    }
}

TEST(ONNXFeConvertException, exception_if_both_unsupported_and_other_translation_exception) {
    const auto path = CommonTestUtils::getModelFromTestModelZoo(
        std::string(ONNX_TEST_MODELS) + "unsupported_ops/unsupported_add_and_incorrect_dts.onnx");
    try {
        convert(path);
        FAIL() << "The exception should be thrown after conversion";
    } catch (const ov::AssertFailure& e) {
        EXPECT_TRUE(std::string{e.what()}.find(
                        "OpenVINO does not support the following ONNX operations: test_domain.UnsupportedAdd") !=
                    std::string::npos);
        EXPECT_TRUE(has_suffix(e.what(), "only 'DCR' and 'CRD' modes are supported\n"));
    } catch (...) {
        FAIL() << "Unexpected exception was thrown";
    }
}

TEST(ONNXFeConvertException, exception_if_both_unsupported_onnx_validation_exception_and_other_exception) {
    const auto path = CommonTestUtils::getModelFromTestModelZoo(
        std::string(ONNX_TEST_MODELS) + "unsupported_ops/unsupported_add_incorrect_dts_and_inst_norm_bad_scale.onnx");
    try {
        convert(path);
        FAIL() << "The exception should be thrown after conversion";
    } catch (const ov::AssertFailure& e) {
        EXPECT_TRUE(std::string{e.what()}.find(
                        "OpenVINO does not support the following ONNX operations: test_domain.UnsupportedAdd") !=
                    std::string::npos);
        EXPECT_TRUE(std::string{e.what()}.find("'stop' input is not a scalar") != std::string::npos);
        EXPECT_TRUE(has_suffix(e.what(), "only 'DCR' and 'CRD' modes are supported\n"));
    } catch (...) {
        FAIL() << "Unexpected exception was thrown";
    }
}
