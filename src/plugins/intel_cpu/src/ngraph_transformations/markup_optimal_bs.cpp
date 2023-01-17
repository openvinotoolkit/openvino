// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "markup_optimal_bs.hpp"
#include "ngraph_transformations/op/fully_connected.hpp"

#include <openvino/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include "rt_info/optimal_batch_size.hpp"
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

NGRAPH_RTTI_DEFINITION(ov::intel_cpu::MarkupOptimalBS, "MarkupOptimalBS", 0);
NGRAPH_RTTI_DEFINITION(ov::intel_cpu::MarkupConvolutionOptimalBS, "MarkupConvolutionOptimalBS", 0);
NGRAPH_RTTI_DEFINITION(ov::intel_cpu::MarkupGroupConvolutionOptimalBS, "MarkupGroupConvolutionOptimalBS", 0);
NGRAPH_RTTI_DEFINITION(ov::intel_cpu::MarkupFullyConnectedOptimalBS, "MarkupFullyConnectedOptimalBS", 0);

namespace {
// TODO: remove this WA
bool conv_with_fused_add(const std::shared_ptr<ov::Node> node) {
    const auto consumers = node->output(0).get_target_inputs();
    if (consumers.size() == 1) {
        const auto consumer = (*consumers.begin()).get_node();
        if (ov::is_type<ov::opset1::Add>(consumer)) {
            const auto second_parent = consumer->get_input_node_shared_ptr(1);
            if (ov::is_type<ov::opset1::Parameter>(second_parent))
                return true;
            if (ov::is_type<ov::opset1::Convert>(second_parent) &&
                ov::is_type<ov::opset1::Parameter>(second_parent->get_input_node_shared_ptr(0)))
                return true;
        }
    }

    return false;
}
}  // namespace

ov::intel_cpu::MarkupConvolutionOptimalBS::MarkupConvolutionOptimalBS() {
    auto conv_m = ov::pass::pattern::wrap_type<ov::opset1::Convolution>(ov::pass::pattern::has_static_shape());

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();

        auto output_shape = m.get_match_value().get_shape();
        const size_t original_batch = output_shape[0];
        output_shape[0] = 1;

        const auto weights_shape = node->get_input_shape(1);
        const auto div = static_cast<double>(ngraph::shape_size(weights_shape)) /
                         static_cast<double>(ngraph::shape_size(output_shape));

        auto get_opt_batch = [&]() -> size_t {
            if (conv_with_fused_add(node))
                return original_batch;

            if (node->get_input_element_type(0).is_real()) {
                return div < 0.32 ? 16 : 1;
            } else {
                if (div < 0.65)
                    return 16;
                else if (div < 1.45)
                    return 2;
                else
                    return 1;
            }
        };

        const size_t new_batch = get_opt_batch();
        const size_t optimal_bs = original_batch % new_batch == 0 ? new_batch : original_batch;
        if (original_batch >= optimal_bs)
            ov::intel_cpu::set_optimal_bs(node, optimal_bs);

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(conv_m, "MarkupConvolutionOptimalBS");
    this->register_matcher(m, callback);
}

ov::intel_cpu::MarkupGroupConvolutionOptimalBS::MarkupGroupConvolutionOptimalBS() {
    auto group_conv_m = ov::pass::pattern::wrap_type<ov::opset1::GroupConvolution>(ov::pass::pattern::has_static_shape());

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();

        auto output_shape = m.get_match_value().get_shape();
        const size_t original_batch = output_shape[0];

        size_t new_batch = original_batch;
        if (node->get_input_element_type(0).is_real()) {
            new_batch = original_batch;
        } else {
            output_shape[0] = 1;
            const auto weights_shape = node->get_input_shape(1);
            const auto div = static_cast<double>(ngraph::shape_size(weights_shape)) /
                             static_cast<double>(ngraph::shape_size(output_shape));
            new_batch = div < 0.34 ? 16 : 1;
        }

        const size_t optimal_bs = original_batch % new_batch == 0 ? new_batch : original_batch;
        if (original_batch >= optimal_bs)
            ov::intel_cpu::set_optimal_bs(node, optimal_bs);

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(group_conv_m, "MarkupGroupConvolutionOptimalBS");
    this->register_matcher(m, callback);
}

ov::intel_cpu::MarkupFullyConnectedOptimalBS::MarkupFullyConnectedOptimalBS() {
    auto group_conv_m = ov::pass::pattern::wrap_type<ov::intel_cpu::FullyConnectedNode>(ov::pass::pattern::has_static_shape());

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        ov::intel_cpu::set_optimal_bs(m.get_match_root(), m.get_match_value().get_shape()[0]);
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(group_conv_m, "MarkupGroupConvolutionOptimalBS");
    this->register_matcher(m, callback);
}

ov::intel_cpu::MarkupOptimalBS::MarkupOptimalBS() {
    add_matcher<MarkupConvolutionOptimalBS>();
    add_matcher<MarkupGroupConvolutionOptimalBS>();
    add_matcher<MarkupFullyConnectedOptimalBS>();
}
