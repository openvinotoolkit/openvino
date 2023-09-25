// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/test_case.hpp"
#include "common_test_utils/test_control.hpp"
#include "gtest/gtest.h"
#include "ngraph/file_util.hpp"
#include "ngraph/ngraph.hpp"
#include "onnx_import/onnx.hpp"
#include "onnx_import/onnx_utils.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START

using namespace ngraph;

static std::string s_manifest = ngraph::file_util::path_join(ov::test::utils::getExecutableDirectory(), "${MANIFEST}");

using Inputs = std::vector<std::vector<float>>;
using Outputs = std::vector<std::vector<float>>;

template <typename OpType, typename DerivedFromNode>
bool matching_node_found_in_graph(const std::vector<DerivedFromNode>& ops,
                                  const std::string& friendly_name,
                                  const std::unordered_set<std::string>& output_names,
                                  int out_tensor_number = 0) {
    return std::any_of(std::begin(ops), std::end(ops), [&](const DerivedFromNode op) {
        if (const std::shared_ptr<OpType> casted = std::dynamic_pointer_cast<OpType>(op)) {
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
        return op->get_friendly_name() == friendly_name && std::dynamic_pointer_cast<OpType>(op) != nullptr;
    });

    if (it != std::end(ops)) {
        return std::dynamic_pointer_cast<OpType>(*it);
    } else {
        return nullptr;
    }
}

OPENVINO_TEST(onnx_tensor_names, simple_model) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(ov::test::utils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/tensor_names.onnx"));

    const auto ops = function->get_ordered_ops();
    EXPECT_TRUE(matching_node_found_in_graph<op::Parameter>(ops, "identity_on_input", {"input", "identity_on_input"}));
    EXPECT_TRUE(matching_node_found_in_graph<op::Relu>(ops, "relu", {"relu_t"}));
    EXPECT_TRUE(matching_node_found_in_graph<op::v0::Abs>(ops, "final_output", {"abs_t", "final_output"}));
    EXPECT_TRUE(matching_node_found_in_graph<op::Result>(function->get_results(),
                                                         "final_output/sink_port_0",
                                                         {"abs_t", "final_output"}));
}

OPENVINO_TEST(onnx_tensor_names, node_multiple_outputs) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(ov::test::utils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/top_k.onnx"));

    const auto ops = function->get_ordered_ops();
    EXPECT_TRUE(matching_node_found_in_graph<op::Parameter>(ops, "x", {"x"}));
    EXPECT_TRUE(matching_node_found_in_graph<ov::op::util::TopKBase>(ops, "indices", {"values"}, 0));
    EXPECT_TRUE(matching_node_found_in_graph<ov::op::util::TopKBase>(ops, "indices", {"indices"}, 1));

    const auto results = function->get_results();
    EXPECT_TRUE(matching_node_found_in_graph<op::Result>(results, "values/sink_port_0", {"values"}));
    EXPECT_TRUE(matching_node_found_in_graph<op::Result>(results, "indices/sink_port_0", {"indices"}));
}

OPENVINO_TEST(onnx_tensor_names, subgraph_with_multiple_nodes) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/max_pool_transposed.onnx"));

    // after the import the 2 Result objects are connected to 2 distinct nodes (MaxPool & Transpose)
    // the original MaxPool operator in the model doesn't have its name set
    const auto ops = function->get_ordered_ops();

    const auto result1 = find_by_friendly_name<op::Result>(ops, "y/sink_port_0");
    EXPECT_NE(result1, nullptr);
    EXPECT_EQ(result1->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "y");

    const auto result2 = find_by_friendly_name<op::Result>(ops, "z/sink_port_0");
    EXPECT_NE(result2, nullptr);
    EXPECT_EQ(result2->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "z");
}

OPENVINO_TEST(onnx_tensor_names, simple_multiout_operator) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(ov::test::utils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/max_pool_simple.onnx"));

    const auto ops = function->get_ordered_ops();

    // in this case both Results are connected directly to the MaxPool node
    const auto result1 = find_by_friendly_name<op::Result>(ops, "y/sink_port_0");
    EXPECT_NE(result1, nullptr);
    EXPECT_EQ(result1->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "z");

    const auto result2 = find_by_friendly_name<op::Result>(ops, "z/sink_port_0");
    EXPECT_NE(result2, nullptr);
    EXPECT_EQ(result2->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "z");
}

OPENVINO_TEST(onnx_tensor_names, simple_multiout_named_operator) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/max_pool_simple_named.onnx"));

    const auto ops = function->get_ordered_ops();

    // in this case both Results are connected directly to the MaxPool node
    const auto result1 = find_by_friendly_name<op::Result>(ops, "y/sink_port_0");
    EXPECT_NE(result1, nullptr);
    EXPECT_EQ(result1->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "z");

    const auto result2 = find_by_friendly_name<op::Result>(ops, "z/sink_port_0");
    EXPECT_NE(result2, nullptr);
    EXPECT_EQ(result2->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "z");
}

OPENVINO_TEST(onnx_tensor_names, subgraph_with_multiple_nodes_named) {
    const auto function = onnx_import::import_onnx_model(file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                                              SERIALIZED_ZOO,
                                                                              "onnx/max_pool_transposed_named.onnx"));

    const auto ops = function->get_ordered_ops();

    const auto result1 = find_by_friendly_name<op::Result>(ops, "y/sink_port_0");
    EXPECT_NE(result1, nullptr);
    EXPECT_EQ(result1->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "y");

    const auto result2 = find_by_friendly_name<op::Result>(ops, "z/sink_port_0");
    EXPECT_NE(result2, nullptr);
    EXPECT_EQ(result2->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "z");
}

OPENVINO_TEST(onnx_tensor_names, subgraph_conv_with_bias) {
    const auto function =
        onnx_import::import_onnx_model(file_util::path_join(ov::test::utils::getExecutableDirectory(),
                                                            SERIALIZED_ZOO,
                                                            "onnx/conv_with_strides_padding_bias.onnx"));

    const auto ops = function->get_ordered_ops();

    const auto result1 = find_by_friendly_name<op::Result>(ops, "D/sink_port_0");
    EXPECT_NE(result1, nullptr);
    EXPECT_EQ(result1->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "D");
    EXPECT_EQ(result1->input(0).get_source_output().get_names(), std::unordered_set<std::string>({"D"}));

    EXPECT_NE(nullptr, find_by_friendly_name<op::v1::Convolution>(ops, "D/WithoutBiases"));
}

OPENVINO_TEST(onnx_tensor_names, subgraph_gemm_with_bias) {
    const auto function = onnx_import::import_onnx_model(
        file_util::path_join(ov::test::utils::getExecutableDirectory(), SERIALIZED_ZOO, "onnx/gemm_abc.onnx"));

    const auto ops = function->get_ordered_ops();

    const auto result1 = find_by_friendly_name<op::Result>(ops, "y/sink_port_0");
    EXPECT_NE(result1, nullptr);
    EXPECT_EQ(result1->input(0).get_source_output().get_node_shared_ptr()->get_friendly_name(), "y");
    EXPECT_EQ(result1->input(0).get_source_output().get_names(), std::unordered_set<std::string>({"y"}));

    EXPECT_NE(nullptr, find_by_friendly_name<op::v0::MatMul>(ops, "y/WithoutBiases"));
}
