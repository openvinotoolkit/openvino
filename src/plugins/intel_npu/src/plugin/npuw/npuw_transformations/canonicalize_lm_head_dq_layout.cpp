// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "canonicalize_lm_head_dq_layout.hpp"

#include "../logging.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace opp = ov::pass::pattern;

namespace {

using ConstPtr = std::shared_ptr<ov::op::v0::Constant>;

ConstPtr get_const(const ov::Output<ov::Node>& output) {
    return ov::as_type_ptr<ov::op::v0::Constant>(output.get_node_shared_ptr());
}

// Const or Reshape(Const) -> the underlying Constant
ConstPtr const_through_reshape(const ov::Output<ov::Node>& output) {
    if (auto cnst = get_const(output)) {
        return cnst;
    }
    if (auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(output.get_node_shared_ptr())) {
        return get_const(reshape->input_value(0));
    }
    return nullptr;
}

// Reshape(x) -> x, otherwise x itself
ov::Output<ov::Node> skip_reshape(const ov::Output<ov::Node>& output) {
    if (auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(output.get_node_shared_ptr())) {
        return reshape->input_value(0);
    }
    return output;
}

// Constant re-viewed with a new shape; shares the data buffer (no copy of the vocab weight).
ConstPtr reshaped(const ConstPtr& cnst, const ov::Shape& shape) {
    return std::make_shared<ov::op::v0::Constant>(*cnst, shape);
}

}  // namespace

namespace ov::npuw {

CanonicalizeLMHeadDQLayout::CanonicalizeLMHeadDQLayout() {
    auto act = opp::any_input();
    auto weight = opp::wrap_type<ov::op::v1::Transpose>();
    auto matmul = opp::wrap_type<ov::op::v0::MatMul>({act, weight});

    auto callback = [=](opp::Matcher& m) {
        auto& pvm = m.get_pattern_value_map();
        auto mm = ov::as_type_ptr<ov::op::v0::MatMul>(pvm.at(matmul).get_node_shared_ptr());
        if (!mm || mm->get_transpose_a() || mm->get_transpose_b()) {
            return false;
        }

        // Transpose [OC,IC] -> [IC,OC]
        auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(mm->input_value(1).get_node_shared_ptr());
        auto perm = transpose ? get_const(transpose->input_value(1)) : nullptr;
        if (!perm || perm->cast_vector<int64_t>() != std::vector<int64_t>{1, 0}) {
            return false;
        }

        // Reshape [OC,1,IC] -> [OC,IC] (or no Reshape if the Multiply is already 2D)
        auto mul_out = skip_reshape(transpose->input_value(0));
        auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(mul_out.get_node_shared_ptr());
        if (!mul || !mul->get_output_partial_shape(0).is_static()) {
            return false;
        }
        const auto mm_w_shape = transpose->get_input_shape(0);  // [OC,IC]
        if (mm_w_shape.size() != 2) {
            return false;
        }
        const size_t OC = mm_w_shape[0];
        const size_t IC = mm_w_shape[1];

        auto sub = ov::as_type_ptr<ov::op::v1::Subtract>(mul->input_value(0).get_node_shared_ptr());
        if (!sub) {
            return false;
        }

        // Weight: Convert(Const u8 [OC,1,IC] | [OC,IC])
        auto cvt_w = ov::as_type_ptr<ov::op::v0::Convert>(sub->input_value(0).get_node_shared_ptr());
        auto qweight = cvt_w ? get_const(cvt_w->input_value(0)) : nullptr;
        if (!qweight || qweight->get_element_type() != ov::element::u8 ||
            ov::shape_size(qweight->get_shape()) != OC * IC) {
            return false;
        }

        // Zero point: [Reshape ->] Convert(Const|Reshape(Const) u8, OC elements)
        auto cvt_z = ov::as_type_ptr<ov::op::v0::Convert>(skip_reshape(sub->input_value(1)).get_node_shared_ptr());
        auto qzerop = cvt_z ? const_through_reshape(cvt_z->input_value(0)) : nullptr;
        if (!qzerop || qzerop->get_element_type() != ov::element::u8 || ov::shape_size(qzerop->get_shape()) != OC) {
            return false;
        }

        // Scale: Const|Reshape(Const) float, OC elements
        auto qscale = const_through_reshape(mul->input_value(1));
        if (!qscale || !qscale->get_element_type().is_real() || ov::shape_size(qscale->get_shape()) != OC) {
            return false;
        }

        LOG_DEBUG("CanonicalizeLMHeadDQLayout: rewriting " << mm->get_friendly_name() << " [OC=" << OC
                                                           << ", IC=" << IC << "]");

        auto new_cvt_w = std::make_shared<ov::op::v0::Convert>(reshaped(qweight, {OC, IC}),
                                                                cvt_w->get_destination_type());
        auto new_cvt_z =
            std::make_shared<ov::op::v0::Convert>(reshaped(qzerop, {OC, 1}), cvt_z->get_destination_type());
        auto new_sub = std::make_shared<ov::op::v1::Subtract>(new_cvt_w, new_cvt_z);
        auto new_mul = std::make_shared<ov::op::v1::Multiply>(new_sub, reshaped(qscale, {OC, 1}));
        auto new_mm = std::make_shared<ov::op::v0::MatMul>(mm->input_value(0), new_mul, false, true);

        new_mm->set_friendly_name(mm->get_friendly_name());
        new_mm->output(0).set_names(mm->output(0).get_names());
        ov::copy_runtime_info(mm, new_mm);
        ov::copy_runtime_info(ov::NodeVector{sub, mul},
                              ov::NodeVector{new_cvt_w, new_cvt_z, new_sub, new_mul});
        ov::replace_node(mm, new_mm);
        return true;
    };

    register_matcher(std::make_shared<opp::Matcher>(matmul, "CanonicalizeLMHeadDQLayout"), std::move(callback));
}

}  // namespace ov::npuw
