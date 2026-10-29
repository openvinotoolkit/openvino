// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_direct_multiply_sin_cos.hpp"

#include <memory>
#include <vector>

#include "openvino/op/cos.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

DisableFP16CompForDirectMultiplySinCos::DisableFP16CompForDirectMultiplySinCos() {
    using namespace ov::pass::pattern;

    auto multiply_lhs = any_input();
    auto multiply_rhs = any_input();
    auto multiply = wrap_type<ov::op::v1::Multiply>({multiply_lhs, multiply_rhs}, type_matches(element::f32));
    // Shape-only ops may sit between the angle and Sin/Cos, e.g. the transpose + flatten of the split RoPE in LTX-2
    // (diffusers LTX2AudioVideoRotaryPosEmbed): Multiply -> Reshape -> Reshape -> Sin/Cos.
    auto layout_1 = optional<ov::op::v1::Reshape, ov::op::v1::Transpose, ov::op::v0::Squeeze, ov::op::v0::Unsqueeze>({multiply, any_input()});
    auto layout_2 = optional<ov::op::v1::Reshape, ov::op::v1::Transpose, ov::op::v0::Squeeze, ov::op::v0::Unsqueeze>({layout_1, any_input()});
    auto sin = wrap_type<ov::op::v0::Sin>({layout_2}, type_matches(element::f32));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto multiply_node = pattern_map.at(multiply).get_node_shared_ptr();
        const auto sin_node = pattern_map.at(sin).get_node_shared_ptr();
        if (transformation_callback(sin_node))
            return false;

        // The angle tensor is whatever feeds Sin; Cos must consume the same tensor
        const auto angle_node = sin_node->get_input_node_shared_ptr(0);
        ov::NodeVector layout_nodes;
        for (const auto& layout : {layout_1, layout_2}) {
            if (pattern_map.count(layout))
                layout_nodes.push_back(pattern_map.at(layout).get_node_shared_ptr());
        }

        std::vector<std::shared_ptr<ov::op::v0::Cos>> cos_nodes;
        for (const auto& user : angle_node->get_users()) {
            if (const auto cos_node = ov::as_type_ptr<ov::op::v0::Cos>(user))
                cos_nodes.push_back(cos_node);
        }
        if (cos_nodes.empty())
            return false;

        ov::disable_conversion(pattern_map.at(multiply_lhs).get_node_shared_ptr(), element::f16);
        ov::disable_conversion(pattern_map.at(multiply_rhs).get_node_shared_ptr(), element::f16);
        ov::disable_conversion(multiply_node, element::f16);
        for (const auto& layout_node : layout_nodes)
            ov::disable_conversion(layout_node, element::f16);
        ov::disable_conversion(sin_node, element::f16);
        for (const auto& cos_node : cos_nodes)
            ov::disable_conversion(cos_node, element::f16);
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sin, "DisableFP16CompForDirectMultiplySinCos");
    register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
