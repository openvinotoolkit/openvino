// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_1d_ops.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "transformations/utils/utils.hpp"

template <class T>
std::shared_ptr<ngraph::Node> convert(const ngraph::Output<ngraph::Node> & data, std::shared_ptr<T> node, ngraph::NodeVector &new_ops);

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

ngraph::matcher_pass_callback get_callback() {
    return [](ngraph::pattern::Matcher& m) {
        auto node = m.get_match_root();
        if (node->input(0).get_partial_shape().rank().get_length() != 3) {
            return false;
        }

        // Insert H dimension equal to 1
        auto input_shape = node->input(0).get_shape();
        auto output_shape = node->output(0).get_shape();

        input_shape.insert(input_shape.begin() + 2, 1);

        ngraph::NodeVector new_ops;

        // Reshape(input_shape)->Op->Reshape(output_shape)
        ngraph::Output<ngraph::Node> last = ngraph::op::util::reshapeTo(node->input_value(0), input_shape);
        last.get_node_shared_ptr()->set_friendly_name(node->get_friendly_name() + "/reshape_begin");
        new_ops.push_back(last.get_node_shared_ptr());

        if (auto max_pool = std::dynamic_pointer_cast<ngraph::opset1::MaxPool>(node)) {
            last = convert(last, max_pool, new_ops);
        } else if (auto avg_pool = std::dynamic_pointer_cast<ngraph::opset1::AvgPool>(node)) {
            last = convert(last, avg_pool, new_ops);
        } else {
            throw ngraph::ngraph_error("Reshape1DOps: op type is not supported");
        }

        last.get_node_shared_ptr()->set_friendly_name(node->get_friendly_name() + "/new");
        new_ops.push_back(last.get_node_shared_ptr());

        last = ngraph::op::util::reshapeTo(last, output_shape);
        last.get_node_shared_ptr()->set_friendly_name(node->get_friendly_name());
        new_ops.push_back(last.get_node_shared_ptr());

        ngraph::copy_runtime_info(node, new_ops);
        node->output(0).replace(last);
        return true;
    };
}

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::Reshape1DAvgPool, "Reshape1DAvgPool", 0);

MKLDNNPlugin::Reshape1DAvgPool::Reshape1DAvgPool() {
    auto pool = ngraph::pattern::wrap_type<ngraph::opset1::AvgPool>(ngraph::pattern::has_static_shape());
    auto m = std::make_shared<ngraph::pattern::Matcher>(pool, "Reshape1DAvgPool");
    this->register_matcher(m, get_callback());
}

NGRAPH_RTTI_DEFINITION(MKLDNNPlugin::Reshape1DMaxPool, "Reshape1DMaxPool", 0);

MKLDNNPlugin::Reshape1DMaxPool::Reshape1DMaxPool() {
    auto pool = ngraph::pattern::wrap_type<ngraph::opset1::MaxPool>(ngraph::pattern::has_static_shape());
    auto m = std::make_shared<ngraph::pattern::Matcher>(pool, "Reshape1DMaxPool");
    this->register_matcher(m, get_callback());
}
