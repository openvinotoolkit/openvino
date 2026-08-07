// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_bf16_comp_cumsum_sin_gen.hpp"

#include <memory>
#include <string>
#include <vector>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_cpu {

namespace {

bool is_l_sin_gen_node(const std::shared_ptr<ov::Node>& node) {
    if (!node) {
        return false;
    }
    return node->get_friendly_name().find("l_sin_gen") != std::string::npos;
}

void mark_fp32_chain(const std::vector<std::shared_ptr<ov::Node>>& nodes) {
    for (const auto& node : nodes) {
        if (node) {
            ov::disable_conversion(node, ov::element::f32, ov::element::bf16);
        }
    }
}

}  // namespace

DisableBF16CompCumSumSinGen::DisableBF16CompCumSumSinGen() {
    MATCHER_SCOPE(DisableBF16CompCumSumSinGen);
    using namespace ov::pass::pattern;

    auto transpose_pre_m = wrap_type<ov::op::v1::Transpose>({any_input(), any_input()});

    auto interp_pre_3_m =
        wrap_type<ov::op::v4::Interpolate, ov::op::v11::Interpolate>({transpose_pre_m, any_input(), any_input()});
    auto interp_pre_4_m = wrap_type<ov::op::v4::Interpolate, ov::op::v11::Interpolate>(
        {transpose_pre_m, any_input(), any_input(), any_input()});
    auto interp_pre_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{interp_pre_3_m, interp_pre_4_m});

    auto transpose1_m = wrap_type<ov::op::v1::Transpose>({interp_pre_m, any_input()});
    auto cumsum_m = wrap_type<ov::op::v0::CumSum>({transpose1_m, any_input()});
    auto mul1_m = wrap_type<ov::op::v1::Multiply>({cumsum_m, any_input()});
    auto transpose2_m = wrap_type<ov::op::v1::Transpose>({mul1_m, any_input()});
    auto mul2_m = wrap_type<ov::op::v1::Multiply>({transpose2_m, any_input()});

    auto interp_down_3_m =
        wrap_type<ov::op::v4::Interpolate, ov::op::v11::Interpolate>({mul2_m, any_input(), any_input()});
    auto interp_down_4_m =
        wrap_type<ov::op::v4::Interpolate, ov::op::v11::Interpolate>({mul2_m, any_input(), any_input(), any_input()});
    auto interp_down_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{interp_down_3_m, interp_down_4_m});

    auto transpose3_m = wrap_type<ov::op::v1::Transpose>({interp_down_m, any_input()});
    auto sin_m = wrap_type<ov::op::v0::Sin>({transpose3_m});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto sin_node = pattern_map.at(sin_m).get_node_shared_ptr();
        if (transformation_callback(sin_node)) {
            return false;
        }

        if (!is_l_sin_gen_node(sin_node)) {
            return false;
        }

        std::vector<std::shared_ptr<ov::Node>> to_mark{
            pattern_map.at(transpose_pre_m).get_node_shared_ptr(),
            pattern_map.at(transpose1_m).get_node_shared_ptr(),
            pattern_map.at(cumsum_m).get_node_shared_ptr(),
            pattern_map.at(mul1_m).get_node_shared_ptr(),
            pattern_map.at(transpose2_m).get_node_shared_ptr(),
            pattern_map.at(mul2_m).get_node_shared_ptr(),
            pattern_map.at(transpose3_m).get_node_shared_ptr(),
            sin_node,
        };

        for (const auto& key : {interp_pre_3_m, interp_pre_4_m, interp_down_3_m, interp_down_4_m}) {
            auto it = pattern_map.find(key);
            if (it != pattern_map.end()) {
                to_mark.push_back(it->second.get_node_shared_ptr());
            }
        }

        mark_fp32_chain(to_mark);

        return true;
    };

    auto m = std::make_shared<Matcher>(sin_m, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_cpu
