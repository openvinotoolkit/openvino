// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_lowered.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/snippets_isa.hpp>
#include "ngraph_functions/builders.hpp"

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> AddFunctionLoweredBroadcast::initLowered() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    std::shared_ptr<Node> add_input0 = nullptr;
    if (!broadcast_shapes[0].empty() && broadcast_shapes[0].back() != input_shapes[0].back()) {
        add_input0 = std::make_shared<ngraph::snippets::op::BroadcastLoad>(data0, broadcast_shapes[0]);
    } else {
        add_input0 = std::make_shared<ngraph::snippets::op::Load>(data0);
    }

    auto data1 = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    std::shared_ptr<Node> add_input1 = nullptr;
    if (!broadcast_shapes[1].empty() && broadcast_shapes[1].back() != input_shapes[1].back()) {
        add_input1 = std::make_shared<ngraph::snippets::op::BroadcastLoad>(data1, broadcast_shapes[1]);
    } else {
        add_input1 = std::make_shared<ngraph::snippets::op::Load>(data1);
    }
    auto add = std::make_shared<op::v1::Add>(add_input0, add_input1);
    auto store = std::make_shared<ngraph::snippets::op::Store>(add);
    return std::make_shared<ov::Model>(NodeVector{store}, ParameterVector{data0, data1});
}
std::shared_ptr<ov::Model> EltwiseThreeInputsLoweredFunction::initLowered() const {
    // todo: implement conversion between std::vector<size_t> and std::vector<Shape>
    auto input_params = ngraph::builder::makeParams(precision, {input_shapes[0], input_shapes[1], input_shapes[2]});
    auto load_or_broadcastload = [&](size_t i) -> std::shared_ptr<Node> {
        // user specified that no broadcasting is required
        if (broadcast_shapes[i].empty()) {
            return std::make_shared<ngraph::snippets::op::Load>(input_params[i]);
        // broadcasting is required: could be Load + BroadcastMove or BroiadcastLoad
        } else {
            // The last dim is processed by vector Tile, so BroadcastLoad is required if the last dim being broadcasted
            if (input_shapes[i].back() == 1 && broadcast_shapes[i].back() != 1) {
                return std::make_shared<ngraph::snippets::op::BroadcastLoad>(input_params[i], broadcast_shapes[i]);
            // Todo: Cover this logics with functional tests, Review FakeBroadcast Emitter
            // Broadcasting of other dims is handled by BroadcastMove. Strictly speaking, broadcasting is achieved via
            // appropriate pointer arithmetics in this case.
            } else {
                auto load = std::make_shared<ngraph::snippets::op::Load>(input_params[i]);
                return std::make_shared<ngraph::snippets::op::BroadcastMove>(load, broadcast_shapes[i]);
            }
        }
    };
    auto add = std::make_shared<op::v1::Add>(load_or_broadcastload(0), load_or_broadcastload(1));

    const std::vector<float> const_values = CommonTestUtils::generate_float_numbers(1, -10., 10.);
    auto sub_scalar = std::make_shared<ngraph::snippets::op::Scalar>(precision, Shape{1}, const_values[0]);
    std::shared_ptr<Node> sub_load;
//  Todo: Uncomment when invalid read in vector tile will be fixed
//    if (input_shapes[2].back() == 1)
//        sub_load = std::make_shared<snippets::op::ScalarLoad>(input_params[2]);
//    else
//        sub_load = std::make_shared<snippets::op::Load>(input_params[2]);
//  remove when the code above is enabled:
    sub_load = std::make_shared<ngraph::snippets::op::Load>(input_params[2]);
    auto sub = std::make_shared<op::v1::Subtract>(sub_load, sub_scalar);
    std::shared_ptr<Node> sub_out;
    if (broadcast_shapes[2].empty())
        sub_out = sub;
    else
        sub_out = std::make_shared<ngraph::snippets::op::BroadcastMove>(sub, broadcast_shapes[2]);
    auto mul = std::make_shared<op::v1::Multiply>(add, sub_out);
    auto store = std::make_shared<ngraph::snippets::op::Store>(mul);
    return std::make_shared<ov::Model>(NodeVector{store}, input_params);
}
}  // namespace snippets
}  // namespace test
}  // namespace ov
