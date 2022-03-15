// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "markup_optimal_bs.hpp"
#include "ngraph_transformations/op/fully_connected.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include "rt_info/optimal_batch_size.hpp"
#include <ngraph/pattern/op/wrap_type.hpp>
#include "transformations/utils/utils.hpp"

NGRAPH_RTTI_DEFINITION(ov::intel_cpu::MarkupOptimalBS, "MarkupOptimalBS", 0);
NGRAPH_RTTI_DEFINITION(ov::intel_cpu::MarkupConvolutionOptimalBS, "MarkupConvolutionOptimalBS", 0);
NGRAPH_RTTI_DEFINITION(ov::intel_cpu::MarkupGroupConvolutionOptimalBS, "MarkupGroupConvolutionOptimalBS", 0);
NGRAPH_RTTI_DEFINITION(ov::intel_cpu::MarkupFullyConnectedOptimalBS, "MarkupFullyConnectedOptimalBS", 0);

ov::intel_cpu::MarkupConvolutionOptimalBS::MarkupConvolutionOptimalBS() {
    auto conv_m = ngraph::pattern::wrap_type<ngraph::opset1::Convolution>(ngraph::pattern::has_static_shape());

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto node = m.get_match_root();

        auto output_shape = m.get_match_value().get_shape();
        const size_t original_batch = output_shape[0];
        output_shape[0] = 1;

        const auto weights_shape = node->get_input_shape(1);
        const auto div = static_cast<double>(ngraph::shape_size(weights_shape)) /
                         static_cast<double>(ngraph::shape_size(output_shape));

        size_t new_batch = original_batch;
        if (node->get_input_element_type(0).is_real()) {
            new_batch = div < 0.32 ? 16 : 1;
        } else {
            if (div < 0.65)
                new_batch = 16;
            else if (div < 1.45)
                new_batch = 2;
            else
                new_batch = 1;
        }

        const size_t optimal_bs = original_batch % new_batch == 0 ? new_batch : original_batch;
        if (original_batch >= optimal_bs)
            ov::intel_cpu::set_optimal_bs(node, optimal_bs);

        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv_m, "MarkupConvolutionOptimalBS");
    this->register_matcher(m, callback);
}

ov::intel_cpu::MarkupGroupConvolutionOptimalBS::MarkupGroupConvolutionOptimalBS() {
    auto group_conv_m = ngraph::pattern::wrap_type<ngraph::opset1::GroupConvolution>(ngraph::pattern::has_static_shape());

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
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

    auto m = std::make_shared<ngraph::pattern::Matcher>(group_conv_m, "MarkupGroupConvolutionOptimalBS");
    this->register_matcher(m, callback);
}

ov::intel_cpu::MarkupFullyConnectedOptimalBS::MarkupFullyConnectedOptimalBS() {
    auto group_conv_m = ngraph::pattern::wrap_type<ov::intel_cpu::FullyConnectedNode>(ngraph::pattern::has_static_shape());

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        ov::intel_cpu::set_optimal_bs(m.get_match_root(), m.get_match_value().get_shape()[0]);
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(group_conv_m, "MarkupGroupConvolutionOptimalBS");
    this->register_matcher(m, callback);
}

ov::intel_cpu::MarkupOptimalBS::MarkupOptimalBS() {
    add_matcher<MarkupConvolutionOptimalBS>();
    add_matcher<MarkupGroupConvolutionOptimalBS>();
    add_matcher<MarkupFullyConnectedOptimalBS>();
}
