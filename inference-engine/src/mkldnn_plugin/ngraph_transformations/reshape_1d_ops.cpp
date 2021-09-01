// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_1d_ops.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph_ops/type_relaxed.hpp>

#include "transformations/utils/utils.hpp"

namespace Reshape1DOps {
template <class BaseOp>
std::shared_ptr<ngraph::Node> convert(const ngraph::Output<ngraph::Node> & data, std::shared_ptr<BaseOp> node, ngraph::NodeVector &new_ops) {
    auto new_strides = node->get_strides();
    auto new_dilations = node->get_dilations();
    auto new_pads_begin = node->get_pads_begin();
    auto new_pad_end = node->get_pads_end();

    new_strides.insert(new_strides.begin(), 1);
    new_dilations.insert(new_dilations.begin(), 1);
    new_pads_begin.insert(new_pads_begin.begin(), 0);
    new_pad_end.insert(new_pad_end.begin(), 0);

    ngraph::Shape new_weights_shape(node->input_value(1).get_shape());
    new_weights_shape.insert(new_weights_shape.begin() + new_weights_shape.size() - 1, 1);
    auto reshape_const = ngraph::opset1::Constant::create(ngraph::element::i64, { new_weights_shape.size() }, new_weights_shape);
    auto weights = ngraph::op::util::make_try_fold<ngraph::opset1::Reshape>(node->input_value(1), reshape_const, true);

    new_ops.push_back(weights);

    if (std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(node)) {
        return std::make_shared<ngraph::op::TypeRelaxed<BaseOp>>(std::vector<ngraph::element::Type>{ngraph::element::f32, ngraph::element::f32},
                                                                 std::vector<ngraph::element::Type>{ngraph::element::f32},
                                                                 ngraph::op::TemporaryReplaceOutputType(data, ngraph::element::f32).get(),
                                                                 ngraph::op::TemporaryReplaceOutputType(weights, ngraph::element::f32).get(),
                                                                 new_strides,
                                                                 new_pads_begin,
                                                                 new_pad_end,
                                                                 new_dilations,
                                                                 node->get_auto_pad());
    } else {
        return std::make_shared<BaseOp>(data,
                                        weights,
                                        new_strides,
                                        new_pads_begin,
                                        new_pad_end,
                                        new_dilations,
                                        node->get_auto_pad());
    }
}

template <>
std::shared_ptr<ngraph::Node> convert(const ngraph::Output<ngraph::Node> & data, std::shared_ptr<ngraph::opset1::MaxPool> node, ngraph::NodeVector & new_ops) {
    auto new_strides = node->get_strides();
    auto new_pads_begin = node->get_pads_begin();
    auto new_pad_end = node->get_pads_end();
    auto new_kernel = node->get_kernel();

    new_strides.insert(new_strides.begin(), 1);
    new_pads_begin.insert(new_pads_begin.begin(), 0);
    new_pad_end.insert(new_pad_end.begin(), 0);
    new_kernel.insert(new_kernel.begin(), 1);

    return std::make_shared<ngraph::opset1::MaxPool>(data,
                                                     new_strides,
                                                     new_pads_begin,
                                                     new_pad_end,
                                                     new_kernel,
                                                     node->get_rounding_type(),
                                                     node->get_auto_pad());
}

template <>
std::shared_ptr<ngraph::Node> convert(const ngraph::Output<ngraph::Node> & data, std::shared_ptr<ngraph::opset1::AvgPool> node, ngraph::NodeVector & new_ops) {
    // Update Pooling attributes with additional dimension
    auto new_strides = node->get_strides();
    auto new_pads_begin = node->get_pads_begin();
    auto new_pad_end = node->get_pads_end();
    auto new_kernel = node->get_kernel();

    new_strides.insert(new_strides.begin(), 1);
    new_pads_begin.insert(new_pads_begin.begin(), 0);
    new_pad_end.insert(new_pad_end.begin(), 0);
    new_kernel.insert(new_kernel.begin(), 1);

    return std::make_shared<ngraph::opset1::AvgPool>(data,
                                             new_strides,
                                             new_pads_begin,
                                             new_pad_end,
                                             new_kernel,
                                             node->get_exclude_pad(),
                                             node->get_rounding_type(),
                                             node->get_auto_pad());
}

std::shared_ptr<ngraph::Node> add_reshape_before(const ngraph::Output<ngraph::Node>& data) {
    auto shape_of = ngraph::op::util::make_try_fold<ngraph::opset1::ShapeOf>(data);

    std::vector<std::int32_t> gather_values_first{ 0, 1 };
    auto gather_indices_first = ngraph::opset1::Constant::create(
        ngraph::element::i64,
        ngraph::Shape{ gather_values_first.size() },
        gather_values_first);

    auto gather_axis = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 0 });
    auto gather_first = ngraph::op::util::make_try_fold<ngraph::opset1::Gather>(shape_of, gather_indices_first, gather_axis);

    std::vector<std::int32_t> gather_values_last{ 2 };
    auto gather_indices_last = ngraph::opset1::Constant::create(
        ngraph::element::i64,
        ngraph::Shape{ gather_values_last.size() },
        gather_values_last);
    auto gather_last = ngraph::op::util::make_try_fold<ngraph::opset1::Gather>(shape_of, gather_indices_last, gather_axis);

    auto unsqueezed_dimension = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 1 });

    ngraph::NodeVector concatInputs = { gather_first, unsqueezed_dimension, gather_last };
    auto concat = ngraph::op::util::make_try_fold<ngraph::opset1::Concat>(concatInputs, 0);
    auto reshape = std::make_shared<ngraph::opset1::Reshape>(data, concat, true);
    return reshape;
}

std::shared_ptr<ngraph::Node> add_reshape_after(const ngraph::Output<ngraph::Node>& data) {
    auto shape_of = ngraph::op::util::make_try_fold<ngraph::opset1::ShapeOf>(data);

    std::vector<std::int32_t> gather_values{ 0, 1, 3 };
    auto gather_const = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ gather_values.size() }, gather_values);
    auto gather_axis = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, { 0 });
    auto gather = ngraph::op::util::make_try_fold<ngraph::opset1::Gather>(shape_of, gather_const, gather_axis);
    auto reshape = std::make_shared<ngraph::opset1::Reshape>(data, gather, true);
    return reshape;
}

ngraph::matcher_pass_callback get_callback() {
    return [](ngraph::pattern::Matcher& m) {
        auto node = m.get_match_root();
        auto input_rank = node->input(0).get_partial_shape().rank();
        if (input_rank.is_dynamic() || input_rank.get_length() != 3) {
            return false;
        }

        ngraph::NodeVector new_ops;

        // Update pshape from [N, C, W] to [N, C, 1, W]
        ngraph::Output<ngraph::Node> last = add_reshape_before(node->input_value(0));
        last.get_node_shared_ptr()->set_friendly_name(node->get_friendly_name() + "/reshape_begin");
        new_ops.push_back(last.get_node_shared_ptr());

        if (auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(node)) {
            last = convert(last, conv, new_ops);
        } else if (auto group_conv = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(node)) {
            last = convert(last, group_conv, new_ops);
        } else if (auto max_pool = std::dynamic_pointer_cast<ngraph::opset1::MaxPool>(node)) {
            last = convert(last, max_pool, new_ops);
        } else if (auto avg_pool = std::dynamic_pointer_cast<ngraph::opset1::AvgPool>(node)) {
            last = convert(last, avg_pool, new_ops);
        } else {
            throw ngraph::ngraph_error("Reshape1DOps: op type is not supported");
        }

        last.get_node_shared_ptr()->set_friendly_name(node->get_friendly_name() + "/new");
        new_ops.push_back(last.get_node_shared_ptr());

        // if convolution is followed by add we need to replace add before output reshape to fuse conv+bias on plug-in side
        std::shared_ptr<ngraph::Node> add_to_replace = nullptr;
        std::shared_ptr<ngraph::Node> reshaped_add = nullptr;
        ngraph::NodeVector bias_ops;
        if (std::dynamic_pointer_cast<ngraph::opset1::Convolution>(node) || std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(node)) {
            ngraph::Shape expected_shape = ngraph::Shape(input_rank.get_length(), 1);
            expected_shape[1] = node->get_output_partial_shape(0)[1].get_length();
            const auto dst_nodes = node->get_output_target_inputs(0);
            if (dst_nodes.size() == 1) {
                add_to_replace = dst_nodes.begin()->get_node()->shared_from_this();
                if (std::dynamic_pointer_cast<ngraph::opset1::Add>(add_to_replace) &&
                    std::dynamic_pointer_cast<ngraph::opset1::Constant>(add_to_replace->get_input_node_shared_ptr(1)) &&
                    add_to_replace->get_input_shape(1) == expected_shape) {
                    ngraph::Shape new_shape(add_to_replace->get_input_shape(1));
                    new_shape.push_back(1);
                    auto new_shape_const = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ new_shape.size() }, new_shape);

                    auto new_bias = ngraph::op::util::make_try_fold<ngraph::opset1::Reshape>(add_to_replace->input_value(1), new_shape_const, true);
                    reshaped_add = std::make_shared<ngraph::opset1::Add>(last, new_bias);
                    reshaped_add->set_friendly_name(add_to_replace->get_friendly_name() + "/new");
                    bias_ops.push_back(new_bias);
                    bias_ops.push_back(reshaped_add);
                }
            }
        }

        if (reshaped_add != nullptr) {
            ngraph::replace_node(node, last.get_node_shared_ptr());
            ngraph::copy_runtime_info(node, new_ops);
            last = reshaped_add;
            node = add_to_replace;
            new_ops = bias_ops;
        }

        // Update pshape from [N, C, 1, W] to [N, C, W]
        last = add_reshape_after(last);
        last.get_node_shared_ptr()->set_friendly_name(node->get_friendly_name());
        ngraph::replace_node(node, last.get_node_shared_ptr());
        ngraph::copy_runtime_info(node, new_ops);
        return true;
    };
}
} // namespace Reshape1DOps

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::Reshape1DConvolution, "Reshape1DConvolution", 0);

MKLDNNPlugin::Reshape1DConvolution::Reshape1DConvolution() {
    auto conv = ngraph::pattern::wrap_type<ngraph::opset1::Convolution>(ngraph::pattern::has_static_rank());
    auto m = std::make_shared<ngraph::pattern::Matcher>(conv, "Reshape1DConvolution");
    this->register_matcher(m, Reshape1DOps::get_callback());
}

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::Reshape1DGroupConvolution, "Reshape1DGroupConvolution", 0);

MKLDNNPlugin::Reshape1DGroupConvolution::Reshape1DGroupConvolution() {
    auto group_conv = ngraph::pattern::wrap_type<ngraph::opset1::GroupConvolution>(ngraph::pattern::has_static_rank());
    auto m = std::make_shared<ngraph::pattern::Matcher>(group_conv, "Reshape1DGroupConvolution");
    this->register_matcher(m, Reshape1DOps::get_callback());
}

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::Reshape1DAvgPool, "Reshape1DAvgPool", 0);

MKLDNNPlugin::Reshape1DAvgPool::Reshape1DAvgPool() {
    auto pool = ngraph::pattern::wrap_type<ngraph::opset1::AvgPool>(ngraph::pattern::has_static_rank());
    auto m = std::make_shared<ngraph::pattern::Matcher>(pool, "Reshape1DAvgPool");
    this->register_matcher(m, Reshape1DOps::get_callback());
}

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::Reshape1DMaxPool, "Reshape1DMaxPool", 0);

MKLDNNPlugin::Reshape1DMaxPool::Reshape1DMaxPool() {
    auto pool = ngraph::pattern::wrap_type<ngraph::opset1::MaxPool>(ngraph::pattern::has_static_rank());
    auto m = std::make_shared<ngraph::pattern::Matcher>(pool, "Reshape1DMaxPool");
    this->register_matcher(m, Reshape1DOps::get_callback());
}
