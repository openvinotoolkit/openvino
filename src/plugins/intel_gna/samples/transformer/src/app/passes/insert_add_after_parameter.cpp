// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "insert_add_after_parameter.hpp"

#include <openvino/cc/pass/itt.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset11.hpp>
#include <openvino/pass/pattern/matcher.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

namespace transformation_sample {
namespace passes {

InsertAddAfterParameter::InsertAddAfterParameter() {
    MATCHER_SCOPE(InsertAddAfterParameter);

    // Add after parameter.
    const auto parameter_op = ov::pass::pattern::wrap_type<ov::opset11::Parameter>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto param = m.get_match_root();
        const size_t index = 0;
        auto consumers = param->output(index).get_target_inputs();

        auto param_shape = param->get_shape();

        auto prodcut = std::accumulate(param_shape.begin(), param_shape.end(), 1, std::multiplies<size_t>());
        std::vector<float> constant_values(prodcut, 1.0);

        auto constant =
            std::make_shared<ov::opset11::Constant>(param->get_element_type(), param_shape, constant_values);

        auto add = std::make_shared<ov::opset11::Add>(param, constant);

        add->set_friendly_name(param->get_friendly_name() + "/add_layer");

        ov::copy_runtime_info(param, add);

        for (auto& consumer : consumers) {
            consumer.replace_source_output(add);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(parameter_op, matcher_name);

    this->register_matcher(m, callback);
}

}  // namespace passes
}  // namespace transformation_sample
