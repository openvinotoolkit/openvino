// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "subgraph_movebroadcast.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "snippets/op/broadcastmove.hpp"
#include "transformations/init_node_info.hpp"

namespace ov {
namespace test {
namespace snippets {

MoveBroadcastFunction::MoveBroadcastFunction(const std::vector<ov::PartialShape>& inputShapes,
                                           ov::element::Type_t precision)
    : SnippetsFunctionBase(inputShapes, precision) {
    OPENVINO_ASSERT(input_shapes.size() == 2, "Got invalid number of input shapes");
}

std::shared_ptr<ov::Model> MoveBroadcastFunction::initOriginal() const {
    auto input0 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[0]);
    auto input1 = std::make_shared<ov::op::v0::Parameter>(precision, input_shapes[1]);
    auto add = std::make_shared<ov::op::v1::Add>(input0, input1);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{input0, input1});

    return model;
}

std::shared_ptr<ov::Model> MoveBroadcastFunction::initReference() const {
    auto ref_data0 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 3});
    auto ref_data1 = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{1, 2, 1});
    auto move1 = std::make_shared<ov::snippets::op::BroadcastMove>(ref_data1, ov::Dimension{3});
    auto add_ref = std::make_shared<ov::op::v1::Add>(ref_data0, move1);
    auto model = std::make_shared<Model>(OutputVector{add_ref}, ParameterVector{ref_data0, ref_data1});

    return model;
}

}  // namespace snippets
}  // namespace test
}  // namespace ov