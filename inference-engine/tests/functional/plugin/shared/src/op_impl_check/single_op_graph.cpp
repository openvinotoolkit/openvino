// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <op_impl_check/op_impl_check.hpp>
#include <op_impl_check/single_op_graph.hpp>

namespace ov {
namespace test {
namespace subgraph {

namespace {
std::shared_ptr<ov::Function> generate(const ov::op::Op &node) {
    return nullptr;
}

// util::BinaryElementwiseArithmetic
std::shared_ptr<ov::Function> generate(const ov::op::v1::Add &node) {
    const auto params = ngraph::builder::makeDynamicParams(ov::element::f32, {{1, 2},
                                                                              {1, 2}});
    const auto softMax = std::make_shared<ov::op::v1::Add>(params.front(), params.back());
    const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(softMax)};
    std::string friendlyName = std::string(node.get_type_info().name) + std::string("_") + node.get_type_info().get_version();
    return std::make_shared<ngraph::Function>(results, params, friendlyName);
}
} // namespace

template <typename T>
std::shared_ptr<ov::Function> generateGraph() {
    T a;
    return generate(a);
}

OpGenerator getOpGeneratorMap() {
    static OpGenerator opGeneratorMap{
#define _OPENVINO_OP_REG(NAME, NAMESPACE) {NAMESPACE::NAME::get_type_info_static(), generateGraph<NAMESPACE::NAME>},
#include "openvino/opsets/opset1_tbl.hpp"
#include "openvino/opsets/opset2_tbl.hpp"
#include "openvino/opsets/opset3_tbl.hpp"
#include "openvino/opsets/opset4_tbl.hpp"
#include "openvino/opsets/opset5_tbl.hpp"
#include "openvino/opsets/opset6_tbl.hpp"
#include "openvino/opsets/opset7_tbl.hpp"
#include "openvino/opsets/opset8_tbl.hpp"
#undef _OPENVINO_OP_REG
    };
    return opGeneratorMap;
}

}  // namespace subgraph
}  // namespace test
}  // namespace ov