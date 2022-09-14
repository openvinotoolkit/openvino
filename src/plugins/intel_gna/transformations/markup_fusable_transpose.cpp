// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <ngraph/opsets/opset9.hpp>
#include <legacy/ngraph_ops/convolution_ie.hpp>
#include <legacy/ngraph_ops/fully_connected.hpp>
#include <legacy/ngraph_ops/scaleshift.hpp>
#include <openvino/cc/ngraph/itt.hpp>
#include <ngraph/rt_info.hpp>
#include <ops/util/util.hpp>
#include <transformations/utils/transformation_helper.hpp>
#include <transformations/rt_info/gna_transpose_fusable.hpp>

#include "markup_fusable_transpose.hpp"

using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::pass::helper;
using namespace ov::intel_gna::ngraph_util;
using namespace ov::intel_gna::rt_info;

namespace {
bool is_skip_operation(const std::shared_ptr<ngraph::Node>& node) {
    return (!std::dynamic_pointer_cast<ngraph::opset9::Transpose>(node) &&
            !std::dynamic_pointer_cast<ngraph::op::FullyConnected>(node) &&
            !std::dynamic_pointer_cast<ngraph::op::ScaleShiftIE>(node) &&
            !std::dynamic_pointer_cast<ngraph::opset9::Result>(node) &&
            (!is_gna_non_functional_node(node) ||
                node->output(0).get_shape().size() == node->input(0).get_shape().size()));
}
} // namespace

bool MarkupFusableTranspose::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(MarkupFusableTranspose);

    for (auto& node : f->get_ordered_ops()) {
        if (!std::dynamic_pointer_cast<ngraph::opset9::Convolution>(node) &&
            !std::dynamic_pointer_cast<ngraph::op::ConvolutionIE>(node)) {
            continue;
        }
        auto in_dims = node->input(0).get_shape();
        auto out_dims = node->output(0).get_shape();

        if (is_one_dim_shapes(in_dims, out_dims)) {
            continue;
        }

        auto current_node = get_next_node_skipping_certain(node, is_skip_operation);
        if (!TransposeOrderMatches(std::dynamic_pointer_cast<ngraph::opset9::Transpose>(current_node), {0, 3, 2, 1})) {
            continue;
        }
        add_transpose_fusable(current_node);
    }

    return false;
}