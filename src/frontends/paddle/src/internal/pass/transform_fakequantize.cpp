// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/pass/transform_fakequantize.hpp"

#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/op/util/op_types.hpp>

#include "default_opset.hpp"
#include "openvino/pass/pattern/op/label.hpp"

using namespace ov::frontend::paddle::op::default_opset;
using namespace ov;
using namespace ov::pass;
using namespace ov::frontend::paddle::op;

/*
                                                                       x
                                                                   |  |  |  |
                                                                   |  |  |  |
                                 /                                 V  v  v  v
                                |              +-----------+     +-----------+
                                |   scale -->  | Multiply  | --> |  Divide   |
    +------------------+        |              +-----------+     +-----------+
    |                  |        |                                      |
    |  quantize_linear |  -->   |                                      v
    |                  |        |                                +-----------+
    +------------------+        |                                |   Round   |
                                 \                               +-----------+                  |   |   |   |
                                                                       |                        v   v   v   v
                                                                       v                    +-------------------+
                                 /                               +-----------+              |                   |
                                |                                |   Clamp   |     === >    |   FakeQuantize    |
                                |                                +-----------+              |                   |
                                |                                      |                    +-------------------+
                                |                                      v                        |   |   |   |
     +------------------+       |                                +-----------+                  v   v   v   v
     |                  |       |                                |  Convert  |
     |  quantize_linear |  -->  |                                +-----------+
     |                  |       |                                      |
     +------------------+       |                                      v
                                |              +-----------+     +-----------+
                                |   scale -->  | Multiply  | --> | Multiply  |
                                |              +-----------+     +-----------+
                                |                                 |  |  |  |
                                 \                                |  |  |  |
                                                                  v  v  v  v
                                                                       Y
*/
ov::frontend::paddle::pass::TransformFakeQuantize::TransformFakeQuantize() {
    // quantize phase
    const auto input_label = ngraph::pattern::any_input();
    const auto q_real_scale_label = ngraph::pattern::wrap_type<Multiply>();
    const auto div_label = ngraph::pattern::wrap_type<Divide>({input_label, q_real_scale_label});
    const auto round_label = ngraph::pattern::wrap_type<Round>({div_label});
    const auto q_clamp_label = ngraph::pattern::wrap_type<Clamp>({round_label});
    // dequantize phase
    const auto dq_cvt_label = ngraph::pattern::wrap_type<Convert>({q_clamp_label});
    const auto dq_real_scale_label = ngraph::pattern::wrap_type<Multiply>();
    const auto output_label = ngraph::pattern::wrap_type<Multiply>({dq_cvt_label, dq_real_scale_label});

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) -> bool {
        const auto& opsMap = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        // prepare for replace
        const auto& output_node = opsMap.at(output_label).get_node_shared_ptr();
        // get the input
        const auto& div_node = opsMap.at(div_label).get_node_shared_ptr();
        if (!div_node->get_input_node_shared_ptr(0)) {
            return false;
        }
        const auto& input_item = div_node->get_input_source_output(0);
        // get the scale
        const auto& scale_node = opsMap.at(q_real_scale_label).get_node_shared_ptr();
        const auto& scale_item = scale_node->get_input_node_shared_ptr(0);
        const auto& scale_value_cast = std::dynamic_pointer_cast<Constant>(scale_item);
        if (!scale_node) {
            return false;
        }
        std::vector<float> scales = scale_value_cast->cast_vector<float>();
        const auto scale = scales[0];
        const auto scale_low = -scale * 128 / 127;
        const auto scale_high = scale;
        const auto input_clamp = std::make_shared<Clamp>(input_item, scale_low, scale_high);
        const auto input_low = std::make_shared<Constant>(element::f32, Shape{1}, scale_low);
        const auto input_high = std::make_shared<Constant>(element::f32, Shape{1}, scale_high);
        const auto output_low = std::make_shared<Constant>(element::f32, Shape{1}, scale_low);
        const auto output_high = std::make_shared<Constant>(element::f32, Shape{1}, scale_high);
        auto fake_node = std::make_shared<FakeQuantize>(input_clamp, input_low, input_high, output_low, output_high, 256);
        fake_node->set_friendly_name(output_node->get_friendly_name());
        replace_node(output_node, fake_node);
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(output_label, "TransformFakeQuantize");
    this->register_matcher(m, callback);
}
