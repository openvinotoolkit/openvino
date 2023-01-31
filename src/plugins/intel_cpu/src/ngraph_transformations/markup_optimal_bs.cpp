// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "markup_optimal_bs.hpp"
#include "ngraph_transformations/op/fully_connected.hpp"

#include <openvino/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include "rt_info/mixed_affinity_props.hpp"
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

using namespace ov::intel_cpu::mixed_affinity;

NGRAPH_RTTI_DEFINITION(MarkupOptimalBS, "MarkupOptimalBS", 0);
NGRAPH_RTTI_DEFINITION(MarkupConvolutionOptimalBS, "MarkupConvolutionOptimalBS", 0);
NGRAPH_RTTI_DEFINITION(MarkupGroupConvolutionOptimalBS, "MarkupGroupConvolutionOptimalBS", 0);
NGRAPH_RTTI_DEFINITION(MarkupFullyConnectedOptimalBS, "MarkupFullyConnectedOptimalBS", 0);
NGRAPH_RTTI_DEFINITION(MarkupBlockers, "MarkupBlockers", 0);

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
        const auto batch_to_set = new_batch <= original_batch ? new_batch : original_batch;
        const size_t n_splits = original_batch / batch_to_set;
        const auto props = Properties(batch_to_set, n_splits);
        set_properties(node, props);

        // set optimal bs also for dequantization subtract before convolution
        // in order to save inseparable Subtract->Convolution sequence after graph partition
        if (node->get_input_element_type(1).is_integral()) {
            auto parent = node->get_input_node_shared_ptr(0);
            if (ov::is_type<opset1::Subtract>(parent) &&
                parent->get_input_element_type(0).is_integral() &&
                parent->get_input_element_type(1).is_integral()) {
                set_properties(parent, props);
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
        const auto batch_to_set = new_batch <= original_batch ? new_batch : original_batch;
        const size_t n_splits = original_batch / batch_to_set;
        const auto props = Properties(batch_to_set, n_splits);
        set_properties(node, props);

        // set optimal bs also for dequantization subtract before convolution
        // in order to save inseparable Subtract->Convolution sequence after graph partition
        if (node->get_input_element_type(1).is_integral()) {
            auto parent = node->get_input_node_shared_ptr(0);
            if (ov::is_type<opset1::Subtract>(parent) &&
                parent->get_input_element_type(0).is_integral() &&
                parent->get_input_element_type(1).is_integral()) {
                set_properties(parent, props);
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
        set_properties(m.get_match_root(), Properties(m.get_match_value().get_shape()[0], 1));
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(group_conv_m, "MarkupGroupConvolutionOptimalBS");
    this->register_matcher(m, callback);
}

MarkupBlockers::MarkupBlockers() {
    auto blocker_m = ov::pass::pattern::wrap_type<ov::opset1::LRN>(ov::pass::pattern::has_static_shape());

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        set_properties(m.get_match_root(), Properties(m.get_match_value().get_shape()[0], 1));
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(blocker_m, "MarkupBlockers");
    this->register_matcher(m, callback);
}

MarkupOptimalBS::MarkupOptimalBS() {
    add_matcher<MarkupConvolutionOptimalBS>();
    add_matcher<MarkupGroupConvolutionOptimalBS>();
    add_matcher<MarkupFullyConnectedOptimalBS>();
    add_matcher<MarkupBlockers>();
}
