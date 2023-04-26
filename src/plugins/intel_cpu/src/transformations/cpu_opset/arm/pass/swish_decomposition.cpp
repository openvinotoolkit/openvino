// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "swish_decomposition.hpp"

#include <openvino/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>

ov::intel_cpu::SwishDecomposition::SwishDecomposition() {
    auto swish = ngraph::pattern::wrap_type<opset4::Swish>();

    ngraph::matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto swish = std::dynamic_pointer_cast<opset4::Swish>(m.get_match_root());
        if (!swish) {
            return false;
        }

        if (swish->input_values().size() == 2) {
            auto beta = std::dynamic_pointer_cast<opset4::Constant>(swish->get_input_node_shared_ptr(1));

            if (!beta || ngraph::shape_size(swish->get_input_shape(1)) != 1) {
                return false;
            }
            auto beta_value = beta->cast_vector<float>()[0];
            if (beta_value != 1.0)
                return false;
        }

        auto input = swish->input_value(0);
        auto sigmoid = std::make_shared<opset4::Sigmoid>(input);
        auto mul = std::make_shared<opset4::Multiply>(input, sigmoid);

        mul->set_friendly_name(swish->get_friendly_name());
        ngraph::copy_runtime_info(swish, {sigmoid, mul});
        ngraph::replace_node(swish, mul);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(swish, "SwishDecomposition");

    register_matcher(m, callback);
}
