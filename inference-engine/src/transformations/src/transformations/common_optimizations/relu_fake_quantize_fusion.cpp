// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/relu_fake_quantize_fusion.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>


NGRAPH_RTTI_DEFINITION(ngraph::pass::ReluFakeQuantizeFusion, "ReluFakeQuantizeFusion", 0);

ngraph::pass::ReluFakeQuantizeFusion::ReluFakeQuantizeFusion() {
    auto data_pattern = ngraph::pattern::any_input();
    auto relu_pattern = ngraph::pattern::wrap_type<opset5::Relu>({data_pattern});
    auto input_low_pattern = ngraph::pattern::wrap_type<opset5::Constant>();
    auto fq_pattern = ngraph::pattern::wrap_type<opset5::FakeQuantize>({relu_pattern, input_low_pattern,
                                                                        ngraph::pattern::wrap_type<opset5::Constant>(),
                                                                        ngraph::pattern::wrap_type<opset5::Constant>(),
                                                                        ngraph::pattern::wrap_type<opset5::Constant>()});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();
        auto data = pattern_map[data_pattern];
        auto relu = pattern_map[relu_pattern];
        auto input_low = pattern_map[input_low_pattern];
        auto input_low_const = std::dynamic_pointer_cast<opset5::Constant>(input_low.get_node_shared_ptr());
        if (!input_low_const)
            return false;
        auto input_low_values = input_low_const->cast_vector<float>();
        bool negative_input_low = std::any_of(input_low_values.begin(), input_low_values.end(), [] (float f) -> bool { return f < 0; });
        if (negative_input_low)
            return false;
        auto fq = std::dynamic_pointer_cast<opset5::FakeQuantize>(pattern_map[fq_pattern].get_node_shared_ptr());
        if (!fq)
            return false;

        auto new_fq = std::make_shared<ngraph::opset5::FakeQuantize>(data,
                                                                     fq->get_input_node_shared_ptr(1),
                                                                     fq->get_input_node_shared_ptr(2),
                                                                     fq->get_input_node_shared_ptr(3),
                                                                     fq->get_input_node_shared_ptr(4),
                                                                     fq->get_levels());
        new_fq->set_friendly_name(fq->get_friendly_name());

        copy_runtime_info({relu.get_node_shared_ptr(), fq}, new_fq);
        replace_node(fq, new_fq);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fq_pattern, "ReluFakeQuantizeFusion");
    this->register_matcher(m, callback);
}
