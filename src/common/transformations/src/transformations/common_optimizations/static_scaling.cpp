// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/static_scaling.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::StaticScaling::StaticScaling() {
    using namespace ov::pass::pattern;
    using ov::pass::pattern::op::Or;

    const float default_scale_factor = 256.f;
    const ov::element::Type infer_prec = ov::element::f32;
    const ov::element::Type scaled_prec = ov::element::f16;

    auto input_m = any_input();
    auto weights_m = wrap_type<ov::op::v0::Constant>(type_matches_any({infer_prec}));
    auto convolution_m = wrap_type<ov::op::v1::Convolution>({ input_m, weights_m });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        OPENVINO_ASSERT(pattern_map.count(convolution_m));

        auto conv = std::dynamic_pointer_cast<ov::op::v1::Convolution>(pattern_map.at(convolution_m).get_node_shared_ptr());
        if (!conv || transformation_callback(conv))
            return false;

        if (conv->get_input_element_type(0) != infer_prec || conv->get_output_element_type(0) != infer_prec) {
            return false;
        }

        auto conv_weight = std::dynamic_pointer_cast<ov::op::v0::Constant>(pattern_map.at(weights_m).get_node_shared_ptr());
        auto conv_weight_convert = std::make_shared<ov::op::v0::Convert>(conv_weight, scaled_prec);
        ov::replace_node(conv_weight, conv_weight_convert);

        auto input = pattern_map.at(input_m);

        ov::Shape scale_const_shape = {1};
        std::vector<float> inverse_scale_value = {(1.f / default_scale_factor)};
        std::shared_ptr<ov::Node> inverse_scale_const = std::make_shared<ov::op::v0::Constant>(infer_prec, scale_const_shape, inverse_scale_value);
        auto scale_down = std::make_shared<ov::op::v1::Multiply>(input.get_node_shared_ptr()->output(0),
                                                                 inverse_scale_const->output(0));
        auto precision_down = std::make_shared<ov::op::v0::Convert>(scale_down, scaled_prec);
        conv->input(0).replace_source_output(precision_down->output(0));

        std::vector<float> scale_value = {default_scale_factor};
        std::shared_ptr<ov::Node> scale_const = std::make_shared<ov::op::v0::Constant>(infer_prec, scale_const_shape, scale_value);
        auto scale_up = std::make_shared<ov::op::v1::Multiply>(conv->output(0),
                                                               scale_const->output(0));
        ov::replace_node(conv, scale_up);
std::cout << "StaticScaling - converted" << std::endl;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(convolution_m, "StaticScaling");
    this->register_matcher(m, callback);
}
