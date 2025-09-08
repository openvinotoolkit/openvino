// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "onnx_utils.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/topk_base.hpp"

using namespace ov;
using namespace ov::frontend::onnx::tests;

static std::string s_manifest = onnx_backend_manifest("${MANIFEST}");

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

template <typename OpType, typename DerivedFromNode>
bool matching_node_found_in_graph(const std::vector<DerivedFromNode>& ops,
                                  const std::string& friendly_name,
                                  const std::unordered_set<std::string>& output_names,
                                  int out_tensor_number = 0) {
    return std::any_of(std::begin(ops), std::end(ops), [&](const DerivedFromNode op) {
        if (const std::shared_ptr<OpType> casted = ov::as_type_ptr<OpType>(op)) {
            const auto& op_friendly_name = casted->get_friendly_name();
            const auto& op_output_names = casted->get_output_tensor(out_tensor_number).get_names();
            if (op_friendly_name == friendly_name && op_output_names == output_names) {
                return true;
            }
        }
        return false;
    });
}

template <typename OpType, typename DerivedFromNode>
std::shared_ptr<OpType> find_by_friendly_name(const std::vector<DerivedFromNode>& ops,
                                              const std::string& friendly_name) {
    const auto it = std::find_if(std::begin(ops), std::end(ops), [&friendly_name](const DerivedFromNode& op) {
        return op->get_friendly_name() == friendly_name && ov::as_type_ptr<OpType>(op) != nullptr;
    });

    if (it != std::end(ops)) {
        return ov::as_type_ptr<OpType>(*it);
    } else {
        return nullptr;
    }
}

OPENVINO_TEST(onnx_tensor_names, simple_model) {
    const auto model = convert_model("tensor_names.onnx");

    const auto ops = model->get_ordered_ops();
    EXPECT_TRUE(
        matching_node_found_in_graph<op::v0::Parameter>(ops, "identity_on_input", {"input", "identity_on_input"}));
    EXPECT_TRUE(matching_node_found_in_graph<op::v0::Relu>(ops, "relu", {"relu_t"}));
    EXPECT_TRUE(matching_node_found_in_graph<op::v0::Abs>(ops, "final_output", {"abs_t", "final_output"}));
    EXPECT_TRUE(matching_node_found_in_graph<op::v0::Result>(model->get_results(),
                                                             "final_output/sink_port_0",
                                                             {"abs_t", "final_output"}));
}

OPENVINO_TEST(onnx_tensor_names, node_multiple_outputs) {
    const auto model = convert_model("top_k.onnx");

    const auto ops = model->get_ordered_ops();
    EXPECT_TRUE(matching_node_found_in_graph<op::v0::Parameter>(ops, "x", {"x"}));
    EXPECT_TRUE(matching_node_found_in_graph<ov::op::util::TopKBase>(ops, "indices", {"values"}, 0));
    EXPECT_TRUE(matching_node_found_in_graph<ov::op::util::TopKBase>(ops, "indices", {"indices"}, 1));

    const auto results = model->get_results();
    EXPECT_TRUE(matching_node_found_in_graph<op::v0::Result>(results, "values/sink_port_0", {"values"}));
    EXPECT_TRUE(matching_node_found_in_graph<op::v0::Result>(results, "indices/sink_port_0", {"indices"}));
}

OPENVINO_TEST(onnx_tensor_names, subgraph_with_multiple_nodes) {
    const auto model = convert_model("max_pool_transposed.onnx");

    // after the import the 2 Result objects are connected to 2 distinct nodes (MaxPool & Transpose)
    // the original MaxPool operator in the model doesn't have its name set
    const auto ops = model->get_ordered_ops();

    const auto result1 = find_by_friendly_name<op::v0::Result>(ops, "y/sink_port_0");
    EXPECT_NE(result1, nullptr);
    EXPECT_EQ(result1->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "y");

    const auto result2 = find_by_friendly_name<op::v0::Result>(ops, "z/sink_port_0");
    EXPECT_NE(result2, nullptr);
    EXPECT_EQ(result2->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "z");
}

OPENVINO_TEST(onnx_tensor_names, simple_multiout_operator) {
    const auto model = convert_model("max_pool_simple.onnx");

    const auto ops = model->get_ordered_ops();

    // in this case both Results are connected directly to the MaxPool node
    const auto result1 = find_by_friendly_name<op::v0::Result>(ops, "y/sink_port_0");
    EXPECT_NE(result1, nullptr);
    EXPECT_EQ(result1->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "z");

    const auto result2 = find_by_friendly_name<op::v0::Result>(ops, "z/sink_port_0");
    EXPECT_NE(result2, nullptr);
    EXPECT_EQ(result2->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "z");
}

OPENVINO_TEST(onnx_tensor_names, simple_multiout_named_operator) {
    const auto model = convert_model("max_pool_simple_named.onnx");

    const auto ops = model->get_ordered_ops();

    // in this case both Results are connected directly to the MaxPool node
    const auto result1 = find_by_friendly_name<op::v0::Result>(ops, "y/sink_port_0");
    EXPECT_NE(result1, nullptr);
    EXPECT_EQ(result1->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "z");

    const auto result2 = find_by_friendly_name<op::v0::Result>(ops, "z/sink_port_0");
    EXPECT_NE(result2, nullptr);
    EXPECT_EQ(result2->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "z");
}

OPENVINO_TEST(onnx_tensor_names, subgraph_with_multiple_nodes_named) {
    const auto model = convert_model("max_pool_transposed_named.onnx");

    const auto ops = model->get_ordered_ops();

    const auto result1 = find_by_friendly_name<op::v0::Result>(ops, "y/sink_port_0");
    EXPECT_NE(result1, nullptr);
    EXPECT_EQ(result1->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "y");

    const auto result2 = find_by_friendly_name<op::v0::Result>(ops, "z/sink_port_0");
    EXPECT_NE(result2, nullptr);
    EXPECT_EQ(result2->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "z");
}

OPENVINO_TEST(onnx_tensor_names, subgraph_conv_with_bias) {
    const auto model = convert_model("conv_with_strides_padding_bias.onnx");

    const auto ops = model->get_ordered_ops();

    const auto result1 = find_by_friendly_name<op::v0::Result>(ops, "D/sink_port_0");
    EXPECT_NE(result1, nullptr);
    EXPECT_EQ(result1->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "D");
    EXPECT_EQ(result1->input(0).get_source_output().get_names(), std::unordered_set<std::string>({"D"}));

    EXPECT_NE(nullptr, find_by_friendly_name<op::v1::Convolution>(ops, "D/WithoutBiases"));
}

OPENVINO_TEST(onnx_tensor_names, subgraph_gemm_with_bias) {
    const auto model = convert_model("gemm_abc.onnx");

    const auto ops = model->get_ordered_ops();

    const auto result1 = find_by_friendly_name<op::v0::Result>(ops, "y/sink_port_0");
    EXPECT_NE(result1, nullptr);
    EXPECT_EQ(result1->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "y");
    EXPECT_EQ(result1->input(0).get_source_output().get_names(), std::unordered_set<std::string>({"y"}));

    EXPECT_NE(nullptr, find_by_friendly_name<op::v0::MatMul>(ops, "y/WithoutBiases"));
}
