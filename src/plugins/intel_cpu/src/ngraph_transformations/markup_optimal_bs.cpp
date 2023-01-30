// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "markup_optimal_bs.hpp"
#include "ngraph_transformations/op/fully_connected.hpp"

#include <openvino/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include "rt_info/optimal_batch_size.hpp"
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

using namespace ov::intel_cpu::mixed_affinity;

NGRAPH_RTTI_DEFINITION(MarkupOptimalBS, "MarkupOptimalBS", 0);
NGRAPH_RTTI_DEFINITION(MarkupConvolutionOptimalBS, "MarkupConvolutionOptimalBS", 0);
NGRAPH_RTTI_DEFINITION(MarkupGroupConvolutionOptimalBS, "MarkupGroupConvolutionOptimalBS", 0);
NGRAPH_RTTI_DEFINITION(MarkupFullyConnectedOptimalBS, "MarkupFullyConnectedOptimalBS", 0);

MarkupConvolutionOptimalBS::MarkupConvolutionOptimalBS() {
    auto conv_m = ov::pass::pattern::wrap_type<ov::opset1::Convolution>(ov::pass::pattern::has_static_shape());

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();

        auto output_shape = m.get_match_value().get_shape();
        const auto weights_shape = node->get_input_shape(1);
        auto metric = static_cast<double>(ov::shape_size(output_shape)) / ov::shape_size(weights_shape);
        const double min_value = 1.;
        const double max_value = 10.;
        const auto original_batch = output_shape[0];

        auto get_opt_batch = [&]() -> size_t {
            auto bs = output_shape[0];
            while ((metric < min_value || metric > max_value) && bs % 2 == 0) {
                if (metric < min_value) {
                    bs *= 2;
                    metric *= 2;
                } else {
                    bs /= 2;
                    metric /= 2;
                }
            }
            return bs;
        };

        const auto new_batch = get_opt_batch();
        // TODO: do we really need this check?
        const auto optimal_bs = original_batch % new_batch == 0 ? new_batch : original_batch;
        const auto batch_to_set = optimal_bs <= original_batch ? optimal_bs : original_batch;
        set_optimal_bs(node, batch_to_set);

        // set optimal bs also for dequantization subtract before convolution
        // in order to save inseparable Subtract->Convolution sequence after graph partition
        if (node->get_input_element_type(1).is_integral()) {
            auto parent = node->get_input_node_shared_ptr(0);
            if (ov::is_type<opset1::Subtract>(parent) &&
                parent->get_input_element_type(0).is_integral() &&
                parent->get_input_element_type(1).is_integral()) {
                set_optimal_bs(parent, batch_to_set);
            }
        }

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(conv_m, "MarkupConvolutionOptimalBS");
    this->register_matcher(m, callback);
}

MarkupGroupConvolutionOptimalBS::MarkupGroupConvolutionOptimalBS() {
    auto group_conv_m = ov::pass::pattern::wrap_type<ov::opset1::GroupConvolution>(ov::pass::pattern::has_static_shape());

        ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();

        auto output_shape = m.get_match_value().get_shape();
        const auto weights_shape = node->get_input_shape(1);
        auto metric = static_cast<double>(ov::shape_size(output_shape)) / ov::shape_size(weights_shape);
        const double min_value = 1.;
        const double max_value = 10.;
        const auto original_batch = output_shape[0];

        auto get_opt_batch = [&]() -> size_t {
            auto bs = output_shape[0];
            while ((metric < min_value || metric > max_value) && bs % 2 == 0) {
                if (metric < min_value) {
                    bs *= 2;
                    metric *= 2;
                } else {
                    bs /= 2;
                    metric /= 2;
                }
            }
            return bs;
        };

        const auto new_batch = get_opt_batch();
        // std::cout << "New bs: " << new_batch << " metric: " << metric << std::endl;
        // TODO: do we really need this check?
        const auto optimal_bs = original_batch % new_batch == 0 ? new_batch : original_batch;
        const auto batch_to_set = optimal_bs <= original_batch ? optimal_bs : original_batch;
        set_optimal_bs(node, batch_to_set);

        // set optimal bs also for dequantization subtract before convolution
        // in order to save inseparable Subtract->Convolution sequence after graph partition
        if (node->get_input_element_type(1).is_integral()) {
            auto parent = node->get_input_node_shared_ptr(0);
            if (ov::is_type<opset1::Subtract>(parent) &&
                parent->get_input_element_type(0).is_integral() &&
                parent->get_input_element_type(1).is_integral()) {
                set_optimal_bs(parent, batch_to_set);
            }
        }

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(group_conv_m, "MarkupGroupConvolutionOptimalBS");
    this->register_matcher(m, callback);
}

MarkupFullyConnectedOptimalBS::MarkupFullyConnectedOptimalBS() {
    auto group_conv_m = ov::pass::pattern::wrap_type<ov::intel_cpu::FullyConnectedNode>(ov::pass::pattern::has_static_shape());

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        set_optimal_bs(m.get_match_root(), m.get_match_value().get_shape()[0]);
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(group_conv_m, "MarkupGroupConvolutionOptimalBS");
    this->register_matcher(m, callback);
}

MarkupOptimalBS::MarkupOptimalBS() {
    add_matcher<MarkupConvolutionOptimalBS>();
    add_matcher<MarkupGroupConvolutionOptimalBS>();
    add_matcher<MarkupFullyConnectedOptimalBS>();
}
