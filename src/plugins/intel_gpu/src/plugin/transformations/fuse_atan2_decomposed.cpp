// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fuse_atan2_decomposed.hpp"

#include "intel_gpu/op/atan2.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/atan.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/select.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

FuseAtan2Decomposed::FuseAtan2Decomposed() {
    using namespace ov::pass::pattern;

    // Atan(div) where div is Divide(lhs, rhs) (frontend form) or
    // Multiply(lhs, Power(rhs, -1)) (post-ConvertDivide form, Multiply is
    // commutative so cover both input orderings).
    auto exp_neg1_m = wrap_type<ov::op::v0::Constant>(
        ov::op::util::constant_predicate<float>([](const std::vector<float>& v) {
            return v.size() == 1 && v[0] == -1.0f;
        }));
    auto power_m = wrap_type<ov::op::v1::Power>({any_input(), exp_neg1_m});
    auto divide_m = wrap_type<ov::op::v1::Divide>({any_input(), any_input()});
    auto mul_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{
        wrap_type<ov::op::v1::Multiply>({any_input(), power_m}),
        wrap_type<ov::op::v1::Multiply>({power_m, any_input()}),
    });
    auto div_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{divide_m, mul_m});
    auto atan_m = wrap_type<ov::op::v0::Atan>({div_m});

    // Sel3 (root) → Sel2(any, atan, Sel1(any, branch_p, branch_m))
    // Both Sel1 branches must consume the same Atan; verified in the callback.
    auto sel1_m = wrap_type<ov::op::v1::Select>({any_input(), any_input(), any_input()});
    auto sel2_m = wrap_type<ov::op::v1::Select>({any_input(), atan_m, sel1_m});
    auto sel3_m = wrap_type<ov::op::v1::Select>({any_input(), any_input(), sel2_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto sel3 = pattern_map.at(sel3_m).get_node_shared_ptr();
        auto sel2 = pattern_map.at(sel2_m).get_node_shared_ptr();
        auto sel1 = pattern_map.at(sel1_m).get_node_shared_ptr();
        auto atan = pattern_map.at(atan_m).get_node_shared_ptr();

        if (transformation_callback(sel3))
            return false;

        // Both branches of Sel1 must consume the Atan (the only structural
        // fingerprint robust to constant folding of the special-case chain).
        auto branch_uses_atan = [&](const std::shared_ptr<ov::Node>& branch) {
            for (size_t i = 0; i < branch->get_input_size(); ++i) {
                if (branch->get_input_node_shared_ptr(i) == atan)
                    return true;
            }
            return false;
        };
        if (!branch_uses_atan(sel1->get_input_node_shared_ptr(1)) ||
            !branch_uses_atan(sel1->get_input_node_shared_ptr(2)))
            return false;

        // Pattern guarantees atan->input(0) is either Divide or Multiply(_, Power(_, -1)).
        ov::Output<ov::Node> lhs, rhs;
        auto div_node = atan->get_input_node_shared_ptr(0);
        if (auto div = ov::as_type_ptr<ov::op::v1::Divide>(div_node)) {
            lhs = div->input_value(0);
            rhs = div->input_value(1);
        } else {
            auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(div_node);
            const size_t pow_idx = ov::is_type<ov::op::v1::Power>(mul->get_input_node_shared_ptr(0)) ? 0 : 1;
            auto pow = ov::as_type_ptr<ov::op::v1::Power>(mul->get_input_node_shared_ptr(pow_idx));
            lhs = mul->input_value(1 - pow_idx);
            rhs = pow->input_value(0);
        }

        const auto& y_et = lhs.get_element_type();
        if (!y_et.is_real() || rhs.get_element_type() != y_et)
            return false;

        auto atan2 = std::make_shared<ov::intel_gpu::op::Atan2>(lhs, rhs);
        atan2->set_friendly_name(sel3->get_friendly_name());
        ov::copy_runtime_info({sel3, sel2, sel1, atan}, atan2);
        ov::replace_node(sel3, atan2);
        return true;
    };

    auto m = std::make_shared<Matcher>(sel3_m, "FuseAtan2Decomposed");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
