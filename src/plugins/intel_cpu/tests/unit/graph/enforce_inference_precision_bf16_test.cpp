// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "graph.h"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/softmax.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

using namespace ov::intel_cpu;

namespace {

const std::string matmul_before_name = "matmul_before";
const std::string cumsum_name = "cumsum";
const std::string softmax_name = "softmax";
const std::string matmul_after_name = "matmul_after";

NodePtr find_graph_node_by_name(const Graph& graph, const std::string& name) {
    for (const auto& node : graph.GetNodes()) {
        if (node->getName() == name || node->getOriginalLayers().find(name) != std::string::npos) {
            return node;
        }
    }
    return nullptr;
}

std::shared_ptr<const ov::Model> create_model_with_tagged_node() {
    const auto shape = ov::Shape{1, 32};
    const auto weights_shape = ov::Shape{32, 32};
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto matmul_before =
        std::make_shared<ov::op::v0::MatMul>(input,
                                             ov::op::v0::Constant::create(ov::element::f32, weights_shape, {1.0f}));
    matmul_before->set_friendly_name(matmul_before_name);

    auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
    auto cumsum = std::make_shared<ov::op::v0::CumSum>(matmul_before, axis);
    cumsum->set_friendly_name(cumsum_name);
    ov::disable_conversion(cumsum, ov::element::f32, ov::element::bf16);

    auto island_softmax = std::make_shared<ov::op::v8::Softmax>(cumsum, 1);
    island_softmax->set_friendly_name(softmax_name);

    auto matmul_after =
        std::make_shared<ov::op::v0::MatMul>(island_softmax,
                                             ov::op::v0::Constant::create(ov::element::f32, weights_shape, {1.0f}));
    matmul_after->set_friendly_name(matmul_after_name);
    auto output = std::make_shared<ov::op::v0::Convert>(matmul_after, ov::element::bf16);

    return std::make_shared<const ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(output)},
                                             ov::ParameterVector{input});
}

void expect_node_precision(const NodePtr& node, const ov::element::Type& precision) {
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->getOriginalInputPrecisionAtPort(0), precision) << node->getName();
    EXPECT_EQ(node->getOriginalOutputPrecisionAtPort(0), precision) << node->getName();
}

}  // namespace

TEST(EnforceInferencePrecisionBF16Test, KeepsFp32IslandBetweenMandatoryBF16Nodes) {
    Config config;
    config.inferencePrecision = ov::element::bf16;
    config.inferencePrecisionSetExplicitly = true;

    auto context = std::make_shared<GraphContext>(config, nullptr, false);
    Graph graph;
    graph.Init(create_model_with_tagged_node(), context);

    expect_node_precision(find_graph_node_by_name(graph, matmul_before_name), ov::element::bf16);
    expect_node_precision(find_graph_node_by_name(graph, cumsum_name), ov::element::f32);
    expect_node_precision(find_graph_node_by_name(graph, softmax_name), ov::element::f32);
    expect_node_precision(find_graph_node_by_name(graph, matmul_after_name), ov::element::bf16);
}
