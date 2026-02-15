// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "onnx_utils.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset11.hpp"

using namespace ov::frontend::onnx::tests;

namespace {
std::shared_ptr<ov::op::util::FrameworkNode> get_framework_node_with_out_name(const std::shared_ptr<ov::Model>& model,
                                                                              const std::string& out_name) {
    for (const auto& op : model->get_ops()) {
        if (auto framework_node = ov::as_type_ptr<ov::op::util::FrameworkNode>(op)) {
            for (const auto& out : op->outputs()) {
                if (out.get_any_name() == out_name) {
                    return framework_node;
                }
            }
        }
    }
    return nullptr;
}
}  // namespace

TEST(ONNXFeConvertPartially, insert_framework_node_if_unsupported) {
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = convert_partially("unsupported_ops/add_unsupported.onnx"));
    ASSERT_TRUE(model);

    EXPECT_EQ(count_ops_of_type<ov::opset11::Add>(model), 1);
    const auto unsupported_add = get_framework_node_with_out_name(model, "Y");
    ASSERT_TRUE(unsupported_add);
    EXPECT_EQ(unsupported_add->get_attrs().get_type_name(), "UnsupportedAdd");
    EXPECT_EQ(unsupported_add->get_attrs().get_opset_name(), "test_domain");
}

TEST(ONNXFeConvertPartially, insert_more_framework_nodes_if_unsupported) {
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = convert_partially("unsupported_ops/two_unsupported_nodes.onnx"));
    ASSERT_TRUE(model);

    EXPECT_EQ(count_ops_of_type<ov::opset11::Add>(model), 1);
    const auto unsupported_add = get_framework_node_with_out_name(model, "X");
    ASSERT_TRUE(unsupported_add);
    EXPECT_EQ(unsupported_add->get_attrs().get_type_name(), "UnsupportedAdd");

    const auto unsupported_abs = get_framework_node_with_out_name(model, "Y_out");
    ASSERT_TRUE(unsupported_abs);
    EXPECT_EQ(unsupported_abs->get_attrs().get_type_name(), "UnsupportedAbs");
}

// validation error - onnx/instance_norm_bad_scale_type.onnx
TEST(ONNXFeConvertPartially, insert_framework_node_if_onnx_validation_exception) {
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = convert_partially("instance_norm_bad_scale_type.onnx"));
    ASSERT_TRUE(model);

    const auto incorrect_instance_norm = get_framework_node_with_out_name(model, "y");
    ASSERT_TRUE(incorrect_instance_norm);
    EXPECT_EQ(incorrect_instance_norm->get_attrs().get_type_name(), "InstanceNormalization");
}

TEST(ONNXFeConvertPartially, insert_framework_node_if_other_translation_exception) {
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = convert_partially("depth_to_space_bad_mode.onnx"));
    ASSERT_TRUE(model);

    const auto incorrect_dts = get_framework_node_with_out_name(model, "B");
    ASSERT_TRUE(incorrect_dts);
    EXPECT_EQ(incorrect_dts->get_attrs().get_type_name(), "DepthToSpace");
}

TEST(ONNXFeConvertPartially, insert_framework_nodes_if_both_unsupported_and_other_translation_exception) {
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = convert_partially("unsupported_ops/unsupported_add_and_incorrect_dts.onnx"));
    ASSERT_TRUE(model);

    EXPECT_EQ(count_ops_of_type<ov::opset11::Abs>(model), 1);
    const auto incorrect_dts = get_framework_node_with_out_name(model, "B");
    ASSERT_TRUE(incorrect_dts);
    EXPECT_EQ(incorrect_dts->get_attrs().get_type_name(), "DepthToSpace");

    const auto unsupported_add = get_framework_node_with_out_name(model, "Y");
    ASSERT_TRUE(unsupported_add);
    EXPECT_EQ(unsupported_add->get_attrs().get_type_name(), "UnsupportedAdd");
}
