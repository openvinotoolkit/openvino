// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "disable_fp16_comp_gated_residual.hpp"

#include <memory>
#include <vector>

#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

DisableFP16CompForQwenImageGatedResidualPattern::DisableFP16CompForQwenImageGatedResidualPattern() {
    using namespace ov::pass;
    using namespace ov::pass::pattern;

    auto outer_split = wrap_type_strict_index<ov::op::v1::VariadicSplit>({any_input(), any_input(), any_input()});
    auto outer_split_out = outer_split->output(0) | outer_split->output(1);
    auto inner_split = wrap_type_strict_index<ov::op::v1::VariadicSplit>({outer_split_out, any_input(), any_input()});
    auto gate = wrap_type<ov::op::v0::Unsqueeze>({inner_split->output(2), any_input()}, type_matches(element::f32));

    auto branch_matmul = wrap_type<ov::op::v0::MatMul>({any_input(), any_input()}, type_matches(element::f32));
    auto linear_add0 = wrap_type<ov::op::v1::Add>({branch_matmul, any_input()}, type_matches(element::f32));
    auto linear_add1 = wrap_type<ov::op::v1::Add>({any_input(), branch_matmul}, type_matches(element::f32));
    auto linear_add = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{linear_add0, linear_add1});

    auto gated_branch0 = wrap_type<ov::op::v1::Multiply>({gate, linear_add}, type_matches(element::f32));
    auto gated_branch1 = wrap_type<ov::op::v1::Multiply>({linear_add, gate}, type_matches(element::f32));
    auto gated_branch = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{gated_branch0, gated_branch1});

    auto residual_add0 = wrap_type<ov::op::v1::Add>({any_input(), gated_branch}, type_matches(element::f32));
    auto residual_add1 = wrap_type<ov::op::v1::Add>({gated_branch, any_input()}, type_matches(element::f32));
    auto residual_add = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{residual_add0, residual_add1});
    auto mvn = wrap_type<ov::op::v6::MVN>({residual_add, any_input()}, type_matches(element::f32));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto mvn_node = pattern_map.at(mvn).get_node_shared_ptr();
        if (transformation_callback(mvn_node))
            return false;

        const std::vector<std::shared_ptr<ov::Node>> pattern_nodes = {
            branch_matmul, linear_add, gate, gated_branch, residual_add, mvn};
        for (const auto& pattern_node : pattern_nodes)
            ov::disable_conversion(pattern_map.at(pattern_node).get_node_shared_ptr(), element::f16);
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mvn, "DisableFP16CompForQwenImageGatedResidualPattern");
    register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
