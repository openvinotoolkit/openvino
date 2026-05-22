// Copyright (C) 2026 Intel Corporation
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

namespace ov::intel_gpu {

namespace {

// Walk Atan(div) → (lhs, rhs). div may be either Divide(lhs, rhs) (frontend
// form) or Multiply(lhs, Power(rhs, -1)) (post-ConvertDivide form). Returns
// false if neither shape is found.
bool extract_atan2_args(const std::shared_ptr<ov::Node>& atan_node,
                        ov::Output<ov::Node>& lhs,
                        ov::Output<ov::Node>& rhs) {
    auto div_node = atan_node->get_input_node_shared_ptr(0);

    if (auto div = ov::as_type_ptr<ov::op::v1::Divide>(div_node)) {
        lhs = div->input_value(0);
        rhs = div->input_value(1);
        return true;
    }

    if (auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(div_node)) {
        for (size_t i = 0; i < 2; ++i) {
            auto pow = ov::as_type_ptr<ov::op::v1::Power>(mul->get_input_node_shared_ptr(i));
            if (!pow)
                continue;
            auto exp_const = ov::as_type_ptr<ov::op::v0::Constant>(pow->get_input_node_shared_ptr(1));
            if (!exp_const)
                continue;
            std::vector<float> exp_val;
            try {
                exp_val = exp_const->cast_vector<float>();
            } catch (...) {
                continue;
            }
            if (exp_val.empty() || exp_val[0] != -1.0f)
                continue;
            lhs = mul->input_value(1 - i);
            rhs = pow->input_value(0);
            return true;
        }
    }
    return false;
}

bool try_fuse_at_select_root(const std::shared_ptr<ov::Node>& sel3_node) {
    auto sel3 = ov::as_type_ptr<ov::op::v1::Select>(sel3_node);
    if (!sel3)
        return false;

    auto sel2 = ov::as_type_ptr<ov::op::v1::Select>(sel3->get_input_node_shared_ptr(2));
    if (!sel2)
        return false;
    auto atan = ov::as_type_ptr<ov::op::v0::Atan>(sel2->get_input_node_shared_ptr(1));
    if (!atan)
        return false;

    auto sel1 = ov::as_type_ptr<ov::op::v1::Select>(sel2->get_input_node_shared_ptr(2));
    if (!sel1)
        return false;

    // Both branches of Sel1 must be Add(atan, ±π).
    auto branch_uses_atan = [&](const std::shared_ptr<ov::Node>& branch) {
        for (size_t i = 0; i < branch->get_input_size(); ++i) {
            if (branch->get_input_node_shared_ptr(i) == atan)
                return true;
        }
        return false;
    };
    auto add_p = sel1->get_input_node_shared_ptr(1);
    auto add_m = sel1->get_input_node_shared_ptr(2);
    if (!branch_uses_atan(add_p) || !branch_uses_atan(add_m))
        return false;

    ov::Output<ov::Node> lhs, rhs;
    if (!extract_atan2_args(atan, lhs, rhs))
        return false;

    const auto& y_et = lhs.get_element_type();
    if (!y_et.is_real())
        return false;
    if (rhs.get_element_type() != y_et)
        return false;

    auto atan2 = std::make_shared<ov::intel_gpu::op::Atan2>(lhs, rhs);
    atan2->set_friendly_name(sel3->get_friendly_name());
    ov::copy_runtime_info({sel3, sel2, sel1, atan}, atan2);
    ov::replace_node(sel3, atan2);
    return true;
}

}  // namespace

bool FuseAtan2Decomposed::run_on_model(const std::shared_ptr<ov::Model>& m) {
    bool changed = false;
    // Snapshot ops first because replace_node mutates the model's op list.
    const auto ops = m->get_ordered_ops();
    for (const auto& node : ops) {
        if (try_fuse_at_select_root(node))
            changed = true;
    }
    return changed;
}

}  // namespace ov::intel_gpu
