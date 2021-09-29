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

std::string Shape2Str(const ngraph::Shape & shape) {
    std::string result;
    for (auto value : shape) {
        result += std::to_string(value);
        result += " ";
    }
    return result;
}

Node CreateTiledConst(Node const_node, const ngraph::Shape & shape) {
    std::shared_ptr<ngraph::opset8::Constant> old_const_node = std::dynamic_pointer_cast<ngraph::opset8::Constant>(const_node);

    const ngraph::Shape const_node_shape = const_node->get_output_tensor(0).get_shape();
    const auto elem_type = old_const_node->get_element_type();
    std::vector<char> new_const_values(elem_type.size() * ngraph::shape_size(shape), 0);

    AxisSet broadcast_axes;
    ngraph::Shape new_const_shape = const_node_shape;
    new_const_shape.insert(new_const_shape.begin(), shape.size() - new_const_shape.size(), 1);
    for (int i = 0; i < new_const_shape.size(); ++i) {
        if (new_const_shape[i] == shape[i])
            continue;
        if (shape[i] % new_const_shape[i]) {
            std::cerr << "incompatible shapes new shape {" << Shape2Str(new_const_shape) << "} target shape {" << Shape2Str(shape) << "}" << std::endl;
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

#ifdef GNA_LEGACY
bool HasConvolutionInput(Node eltwise_node) {
    const auto left_parent_node = eltwise_node->input_value(0).get_node_shared_ptr();
    const auto right_parent_node = eltwise_node->input_value(1).get_node_shared_ptr();

    return (IsAnyOfLayerTypes<ngraph::op::ConvolutionIE,
                               ngraph::opset8::Convolution>(left_parent_node) ||
            IsAnyOfLayerTypes<ngraph::op::ConvolutionIE,
                               ngraph::opset8::Convolution>(right_parent_node));
}

bool IsFusedBiasConst(Node const_node) {
    const ngraph::Shape & const_out_shape = const_node->get_output_tensor(0).get_shape();

    return (const_out_shape.size() == 4 &&
            const_out_shape[0] == 1 &&
            const_out_shape[2] == 1 &&
            const_out_shape[3] == 1);
}
#endif

bool HasDynamicShape(Node node) {
    const auto & shape = node->get_output_partial_shape(0);
    return (shape.rank().is_dynamic() || shape[0].is_dynamic());
}

bool DoTransformation(Node const_node, Node eltwise_node) {
    if (HasDynamicShape(const_node) || HasDynamicShape(eltwise_node))
        return false;

#ifdef GNA_LEGACY
    /* That is work around the problem with the next transformations
        ngraph::pass::ConvertOpSet1ToLegacy -> ngraph::pass::BiasFusions ->
                                                    ngraph::pass::ConvAddFusion, ngraph::pass::ConvMultiplyFusion
        That transormations fuse bias into convolution and recognizes const node as [1, C, 1, 1].
        TODO: remove that work around after removing ConvertOpSet1ToLegacy transormations
    */
    if (HasConvolutionInput(eltwise_node) && IsFusedBiasConst(const_node))
        return false;
#endif
    const ngraph::Shape & eltwise_out_shape = eltwise_node->get_output_tensor(0).get_shape();
    const size_t eltwise_out_dims_product = ngraph::shape_size(eltwise_out_shape);

    {
        const ngraph::Shape & const_out_shape = const_node->get_output_tensor(0).get_shape();
        const size_t const_out_dims_product = ngraph::shape_size(const_out_shape);

        if (const_out_shape.size() > eltwise_out_shape.size() ||
            const_out_dims_product == eltwise_out_dims_product ||
            eltwise_out_dims_product % const_out_dims_product) {
            return false;
        }
    }

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
    auto eltwise = ngraph::pattern::wrap_type<ngraph::opset8::Add,
                                    ngraph::opset8::Subtract,
                                    ngraph::opset8::Multiply,
                                    ngraph::op::Eltwise>({eltwise_input, ngraph::pattern::any_input()});

     ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto const_node = pattern_map.at(constant).get_node_shared_ptr();
        auto eltwise_node = pattern_map.at(eltwise).get_node_shared_ptr();

        return DoTransformation(const_node, eltwise_node);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(eltwise, matcher_name);
    this->register_matcher(m, callback);
}
