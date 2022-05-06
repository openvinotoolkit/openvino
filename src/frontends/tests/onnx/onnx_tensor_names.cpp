// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "engines_util/test_case.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "onnx_import/onnx.hpp"
#include "onnx_import/onnx_utils.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;

static std::string s_manifest = "${MANIFEST}";

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

NGRAPH_TEST(onnx_tensor_names, simple_model) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/tensor_names.onnx"));

    const auto ops = function->get_ordered_ops();
    EXPECT_TRUE(matching_node_found_in_graph<op::Parameter>(ops, "identity_on_input", {"input", "identity_on_input"}));
    EXPECT_TRUE(matching_node_found_in_graph<op::Relu>(ops, "relu_t", {"relu_t"}));
    EXPECT_TRUE(matching_node_found_in_graph<op::v0::Abs>(ops, "final_output", {"abs_t", "final_output"}));
    EXPECT_TRUE(matching_node_found_in_graph<op::Result>(function->get_results(),
                                                         "final_output/sink_port_0",
                                                         {"abs_t", "final_output"}));
}

NGRAPH_TEST(onnx_tensor_names, node_multiple_outputs) {
    auto function = onnx_import::import_onnx_model(file_util::path_join(SERIALIZED_ZOO, "onnx/top_k.onnx"));

    const auto ops = function->get_ordered_ops();
    EXPECT_TRUE(matching_node_found_in_graph<op::Parameter>(ops, "x", {"x"}));
    EXPECT_TRUE(matching_node_found_in_graph<op::v1::TopK>(ops, "indices", {"values"}, 0));
    EXPECT_TRUE(matching_node_found_in_graph<op::v1::TopK>(ops, "indices", {"indices"}, 1));

    const auto results = function->get_results();
    EXPECT_TRUE(matching_node_found_in_graph<op::Result>(results, "values/sink_port_0", {"values"}));
    EXPECT_TRUE(matching_node_found_in_graph<op::Result>(results, "indices/sink_port_1", {"indices"}));
}
