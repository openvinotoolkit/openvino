// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/broadcast_const.hpp"

#include "transformations/utils/transformation_helper.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "legacy/ngraph_ops/eltwise.hpp"
#include <transformations/utils/utils.hpp>

#include <vector>
#include <functional>
#include <numeric>

using namespace GNAPluginNS;

using Node = std::shared_ptr<ngraph::Node>;
using Nodes = std::vector<Node>;
using Input = ngraph::Input<ngraph::Node>;
using Inputs = std::vector<Input>;
using AxisSet = std::set<size_t>;

namespace {

bool HasDynamicShape(Node node) {
    const auto & shape = node->get_output_partial_shape(0);
    return shape.is_dynamic();
}

/*
 * really returns only NUMPY and PDPD
 * since another types are filtered out in matcher pattern
 */

ov::op::BroadcastModeSpec GetBroadcastType(Node eltwise_node) {
    auto node = std::dynamic_pointer_cast<ov::op::util::BinaryElementwiseArithmetic>(eltwise_node);

    // check if it's an ngraph::op::Eltwise layer without broadcast type info
    if (!node)
        return ov::op::BroadcastType::NUMPY;

    switch (node->get_autob().m_type) {
        case ov::op::AutoBroadcastType::NUMPY:
            return ov::op::BroadcastType::NUMPY;
        case ov::op::AutoBroadcastType::PDPD:
            return ov::op::BroadcastType::PDPD;
        default:
            return ov::op::BroadcastType::NONE;
    }

    return ov::op::BroadcastType::NONE;
}

bool DoTransformation(Node const_node, Node eltwise_node) {
    if (HasDynamicShape(const_node) || HasDynamicShape(eltwise_node))
        return false;
    const ngraph::Shape & eltwise_out_shape = eltwise_node->get_output_tensor(0).get_shape();

    auto broadcast_const = ngraph::opset8::Constant::create(ngraph::element::Type_t::i64,
                                         ngraph::Shape{eltwise_out_shape.size()}, eltwise_out_shape);

    auto new_const_node = ngraph::op::util::make_try_fold<ngraph::opset8::Broadcast>(const_node,
                                                                                     broadcast_const,
                                                                                     GetBroadcastType(eltwise_node));

    ngraph::replace_node(const_node, new_const_node);
    return true;
}

/*
 * Do not transform graph with NONE/EXPLICIT broadcast modes eltwise layer
 * since that types are not broadcastable at all
 */
bool IsEltwiseAcceptable(const ngraph::Output<ngraph::Node>& output) {
    auto node = std::dynamic_pointer_cast<ov::op::util::BinaryElementwiseArithmetic>(output.get_node_shared_ptr());
    if (!node)
        return true;

    const ov::op::AutoBroadcastType type = node->get_autob().m_type;
    return (type == ov::op::AutoBroadcastType::NUMPY || type == ov::op::AutoBroadcastType::PDPD);
}

} // namespace


BroadcastAddMultiplyConst::BroadcastAddMultiplyConst() {
    MATCHER_SCOPE(BroadcastAddMultiplyConst);

    auto constant = ngraph::pattern::wrap_type<ngraph::opset8::Constant>();
    auto fake_quantize = ngraph::pattern::wrap_type<ngraph::opset8::FakeQuantize>({constant,
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset8::Constant>()});
    auto eltwise_input = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{constant, fake_quantize});
    auto eltwise_left_const = ngraph::pattern::wrap_type<ngraph::opset8::Add,
                                    ngraph::opset8::Subtract,
                                    ngraph::opset8::Multiply,
                                    ngraph::op::Eltwise>({eltwise_input, ngraph::pattern::any_input()}, IsEltwiseAcceptable);
    auto eltwise_right_const = ngraph::pattern::wrap_type<ngraph::opset8::Add,
                                    ngraph::opset8::Subtract,
                                    ngraph::opset8::Multiply,
                                    ngraph::op::Eltwise>({ngraph::pattern::any_input(), eltwise_input}, IsEltwiseAcceptable);
    auto eltwise = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{eltwise_left_const, eltwise_right_const});

     ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto const_node = pattern_map.at(constant).get_node_shared_ptr();

        auto eltwise_node_it = pattern_map.find(eltwise_left_const);
        if (eltwise_node_it == pattern_map.end())
            eltwise_node_it = pattern_map.find(eltwise_right_const);
        if (eltwise_node_it == pattern_map.end())
            return false;

        return DoTransformation(const_node, eltwise_node_it->second.get_node_shared_ptr());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(eltwise, matcher_name);
    this->register_matcher(m, callback);
}
