// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "canonicalize_lm_head_dq_layout.hpp"

#include "../logging.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
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

    // Static shapes only; '?' is a plain wildcard, no symbols are bound or required.
    auto weight_m = wrap_type<v0::Constant>(type_matches(element::u8) && shape_matches("?, 1, ?"));
    auto cvt_w_m = wrap_type<v0::Convert>({weight_m});

    // The zero point is reshaped [d] -> [d,1,1] either before or after the Convert.
    auto zerop_m = wrap_type<v0::Constant>(type_matches(element::u8) && rank_equals(1));
    auto zerop_rs_pre_m = wrap_type<v1::Reshape>({zerop_m, wrap_const()}, shape_matches("?, 1, 1"));
    auto cvt_z_pre_m = wrap_type<v0::Convert>({zerop_rs_pre_m});
    auto cvt_z_post_m = wrap_type<v0::Convert>({zerop_m});
    auto zerop_rs_post_m = wrap_type<v1::Reshape>({cvt_z_post_m, wrap_const()}, shape_matches("?, 1, 1"));
    auto sub_m = wrap_type<v1::Subtract>({cvt_w_m, cvt_z_pre_m | zerop_rs_post_m});

    auto scale_m = wrap_type<v0::Constant>(type_matches_any({element::f16, element::f32}) && rank_equals(1));
    auto scale_rs_m = wrap_type<v1::Reshape>({scale_m, wrap_const()}, shape_matches("?, 1, 1"));
    auto mul_m = wrap_type<v1::Multiply>({sub_m, scale_rs_m});

    // Reshape [OC,1,IC] -> [OC,IC]: drops the unit middle dimension, nothing else.
    const auto drops_unit_dim = [](const Output<Node>& out) {
        const auto& in_shape = out.get_node()->get_input_partial_shape(0);
        const auto& out_shape = out.get_partial_shape();
        return in_shape.is_static() && out_shape.is_static() && in_shape.size() == 3 && out_shape.size() == 2 &&
               in_shape[1] == 1 && in_shape[0] == out_shape[0] && in_shape[2] == out_shape[1];
    };
    auto weight_rs_m = wrap_type<v1::Reshape>({mul_m, wrap_const()}, ov::pass::pattern::op::Predicate(drops_unit_dim));
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
            auto new_cnst = std::make_shared<v0::Constant>(*cnst, shape);
            ov::copy_runtime_info(cnst, new_cnst);
            // WeightlessCacheAttribute is not copyable, so copy_runtime_info skips it. NPUW needs it
            // to reference the weights by their file offset; without it the whole tensor is copied
            // (~389 MB for the LM head). The data layout is unchanged, so the attribute stays valid.
            const auto key = ov::WeightlessCacheAttribute::get_type_info_static();
            const auto& rt_info = cnst->get_rt_info();
            if (const auto it = rt_info.find(key); it != rt_info.end()) {
                new_cnst->get_rt_info()[key] = it->second;
            }
            return new_cnst;
        };

        // Subtract requires equal types of its inputs, so both Converts target the Subtract's type.
        const auto dq_type = sub->get_output_element_type(0);
        auto new_cvt_w = std::make_shared<v0::Convert>(reshaped(qweight, {OC, IC}), dq_type);
        auto new_cvt_z = std::make_shared<v0::Convert>(reshaped(qzerop, {qzerop->get_shape()[0], 1}), dq_type);
        auto new_sub = std::make_shared<v1::Subtract>(new_cvt_w, new_cvt_z);
        auto new_mul = std::make_shared<v1::Multiply>(new_sub, reshaped(qscale, {qscale->get_shape()[0], 1}));
        auto new_matmul = std::make_shared<v0::MatMul>(matmul->input_value(0), new_mul, false, true);

        ov::copy_runtime_info(ov::NodeVector{sub, mul}, ov::NodeVector{new_cvt_w, new_cvt_z, new_sub, new_mul});
        // Moves the tensor names and runtime info; also the friendly name when the MatMul feeds a Result.
        return ov::replace_output_update_name(matmul->output(0), new_matmul->output(0));
    };

    register_matcher(std::make_shared<Matcher>(matmul_m, "CanonicalizeLMHeadDQLayout"), std::move(callback));
}

}  // namespace ov::npuw
