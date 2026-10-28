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
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::npuw {

CanonicalizeLMHeadDQLayout::CanonicalizeLMHeadDQLayout() {
    using namespace ov::pass::pattern;
    using ov::pass::operator|;
    namespace v0 = ov::op::v0;
    namespace v1 = ov::op::v1;

    // The symbols OC / IC are shared by all the predicates below, so the pattern also enforces
    // that weight, zero point and scale are per-channel along the same OC dimension.
    auto weight_m = wrap_type<v0::Constant>(type_matches(element::u8) && shape_matches("OC, 1, IC"));
    auto cvt_w_m = wrap_type<v0::Convert>({weight_m});

    // The zero point is reshaped to [OC,1,1] either before or after the Convert.
    auto zerop_m = wrap_type<v0::Constant>(type_matches(element::u8) && shape_matches("OC"));
    auto zerop_rs_pre_m = wrap_type<v1::Reshape>({zerop_m, wrap_const()}, shape_matches("OC, 1, 1"));
    auto cvt_z_pre_m = wrap_type<v0::Convert>({zerop_rs_pre_m});
    auto cvt_z_post_m = wrap_type<v0::Convert>({zerop_m});
    auto zerop_rs_post_m = wrap_type<v1::Reshape>({cvt_z_post_m, wrap_const()}, shape_matches("OC, 1, 1"));
    auto sub_m = wrap_type<v1::Subtract>({cvt_w_m, cvt_z_pre_m | zerop_rs_post_m});

    auto scale_m = wrap_type<v0::Constant>(type_matches_any({element::f16, element::f32}) && shape_matches("OC"));
    auto scale_rs_m = wrap_type<v1::Reshape>({scale_m, wrap_const()}, shape_matches("OC, 1, 1"));
    auto mul_m = wrap_type<v1::Multiply>({sub_m, scale_rs_m});

    auto weight_rs_m = wrap_type<v1::Reshape>({mul_m, wrap_const()}, shape_matches("OC, IC"));
    auto transpose_m = wrap_type<v1::Transpose>({weight_rs_m, wrap_type<v0::Constant>(value_matches("1, 0"))});
    auto matmul_m =
        wrap_type<v0::MatMul>({any_input(), transpose_m}, {{"transpose_a", false}, {"transpose_b", false}});

    auto callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto matmul = pattern_map.at(matmul_m).get_node_shared_ptr();
        const auto sub = pattern_map.at(sub_m).get_node_shared_ptr();
        const auto mul = pattern_map.at(mul_m).get_node_shared_ptr();
        const auto qweight = ov::as_type_ptr<v0::Constant>(pattern_map.at(weight_m).get_node_shared_ptr());
        const auto qzerop = ov::as_type_ptr<v0::Constant>(pattern_map.at(zerop_m).get_node_shared_ptr());
        const auto qscale = ov::as_type_ptr<v0::Constant>(pattern_map.at(scale_m).get_node_shared_ptr());

        const size_t OC = qweight->get_shape()[0];
        const size_t IC = qweight->get_shape()[2];
        LOG_DEBUG("CanonicalizeLMHeadDQLayout: rewriting " << matmul->get_friendly_name() << " [OC=" << OC
                                                           << ", IC=" << IC << "]");

        // Re-view the compressed Constants with the new shape: shares the data buffer, which
        // matters for the vocab-sized weight.
        const auto reshaped = [](const std::shared_ptr<v0::Constant>& cnst, const ov::Shape& shape) {
            return std::make_shared<v0::Constant>(*cnst, shape);
        };

        // Subtract requires equal types of its inputs, so both Converts target the Subtract's type.
        const auto dq_type = sub->get_output_element_type(0);
        auto new_cvt_w = std::make_shared<v0::Convert>(reshaped(qweight, {OC, IC}), dq_type);
        auto new_cvt_z = std::make_shared<v0::Convert>(reshaped(qzerop, {OC, 1}), dq_type);
        auto new_sub = std::make_shared<v1::Subtract>(new_cvt_w, new_cvt_z);
        auto new_mul = std::make_shared<v1::Multiply>(new_sub, reshaped(qscale, {OC, 1}));
        auto new_matmul = std::make_shared<v0::MatMul>(matmul->input_value(0), new_mul, false, true);

        new_matmul->set_friendly_name(matmul->get_friendly_name());
        new_matmul->output(0).set_names(matmul->output(0).get_names());
        ov::copy_runtime_info(matmul, new_matmul);
        ov::copy_runtime_info(ov::NodeVector{sub, mul}, ov::NodeVector{new_cvt_w, new_cvt_z, new_sub, new_mul});
        ov::replace_node(matmul, new_matmul);
        return true;
    };

    register_matcher(std::make_shared<Matcher>(matmul_m, "CanonicalizeLMHeadDQLayout"), std::move(callback));
}

}  // namespace ov::npuw
