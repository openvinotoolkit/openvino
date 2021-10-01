// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/cc/ngraph/itt.hpp>

#include "transformations/broadcast_const.hpp"

#include "transformations/utils/transformation_helper.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "legacy/ngraph_ops/eltwise.hpp"
#include "ngraph_ops/convolution_ie.hpp"
#include "ngraph/runtime/reference/broadcast.hpp"

#include <vector>
#include <functional>
#include <numeric>

using namespace GNAPluginNS;

NGRAPH_RTTI_DEFINITION(BroadcastConst, "BroadcastConst", 0);

using Node = std::shared_ptr<ngraph::Node>;
using Nodes = std::vector<Node>;
using Input = ngraph::Input<ngraph::Node>;
using Inputs = std::vector<Input>;
using AxisSet = std::set<size_t>;

namespace {

Node CreateTiledConst(Node const_node, const ngraph::Shape & shape) {
    std::shared_ptr<ngraph::opset8::Constant> old_const_node = std::dynamic_pointer_cast<ngraph::opset8::Constant>(const_node);

    const ngraph::Shape const_node_shape = const_node->get_output_tensor(0).get_shape();
    const auto elem_type = old_const_node->get_element_type();
    std::vector<char> new_const_values(elem_type.size() * ngraph::shape_size(shape), 0);

    AxisSet broadcast_axes;
    ngraph::Shape new_const_shape = const_node_shape;
    new_const_shape.insert(new_const_shape.begin(), shape.size() - new_const_shape.size(), 1);
    for (size_t i = 0; i < new_const_shape.size(); ++i) {
        if (new_const_shape[i] == shape[i])
            continue;
        if (shape[i] % new_const_shape[i]) {
            return {};
        }
        broadcast_axes.insert(i);
    }

    ngraph::runtime::reference::broadcast(static_cast<const char*>(old_const_node->get_data_ptr()),
                        new_const_values.data(),
                        new_const_shape,
                        shape,
                        broadcast_axes,
                        elem_type.size());

    return ngraph::opset8::Constant::create(old_const_node->get_output_element_type(0),
                                            shape,
                                            new_const_values.data());
}

bool HasDynamicShape(Node node) {
    const auto & shape = node->get_output_partial_shape(0);
    return shape.is_dynamic();
}

bool DoTransformation(Node const_node, Node eltwise_node) {

    if (HasDynamicShape(const_node) || HasDynamicShape(eltwise_node))
        return false;

    const ngraph::Shape & eltwise_out_shape = eltwise_node->get_output_tensor(0).get_shape();
    auto new_const_node = CreateTiledConst(const_node, eltwise_out_shape);

    if (!new_const_node)
        return false;

    ngraph::replace_node(const_node, new_const_node);

    return true;
}

} // namespace

/**
 * @brief Cannot use precise ngraph pattern since we can have arbitary number of non-functional
 * nodes between Const, FakeQuantize and Eltwise layers
 */

BroadcastConst::BroadcastConst() {
    MATCHER_SCOPE(BroadcastConst);

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
                                    ngraph::op::Eltwise>({eltwise_input, ngraph::pattern::any_input()});
    auto eltwise_right_const = ngraph::pattern::wrap_type<ngraph::opset8::Add,
                                    ngraph::opset8::Subtract,
                                    ngraph::opset8::Multiply,
                                    ngraph::op::Eltwise>({ngraph::pattern::any_input(), eltwise_input});
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
