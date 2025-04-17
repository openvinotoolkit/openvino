// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/sdpa_fusion.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace pass {

SDPAFusion::SDPAFusion() {
    using namespace ov::pass::pattern;
    using namespace ov::gen_pattern;

    auto q_base = makePattern(ov::Rank(4));
    auto q_shape = any_input();
    auto q = optional<ov::op::v1::Reshape>({q_base, q_shape});

    auto k_base = makePattern(ov::Rank(4));
    auto k_shape = any_input();
    auto k = optional<ov::op::v1::Reshape>({k_base, k_shape});

    auto v_base = makePattern(ov::Rank(4));
    auto v_proj_shape_m = any_input();
    auto v = optional<ov::op::v1::Reshape>({v_base, v_proj_shape_m});

    // Optional k scale
    auto attn_scale = any_input();
    // K optional transpose + optional scale
    auto k_trans_dims = any_input();
    auto k_opt_transposed = optional<ov::op::v1::Transpose>({k, k_trans_dims});
    auto k_opt_transposed_scaled = wrap_type<ov::op::v1::Multiply>({k_opt_transposed, attn_scale});
    auto k_opt_transposed_opt_scaled = k_opt_transposed_scaled | k_opt_transposed;

    // No transpose check here, there are scenarios where k is not transposed and that uses equation (A*B)^T = B^T * A^T
    auto qk = wrap_type<ov::op::v0::MatMul>({q, k_opt_transposed_opt_scaled});

    // Optional unsqueeze that is converted to Reshape
    auto unsqueeze_axis = wrap_type<ov::op::v0::Constant>();
    auto qk_opt_unsqueeze = optional<ov::op::v1::Reshape>({qk, unsqueeze_axis});

    auto qk_scaled = wrap_type<ov::op::v1::Multiply>({qk_opt_unsqueeze, attn_scale});
    auto qk_opt_scaled = qk_scaled | qk_opt_unsqueeze;

    // optional mask add, there are patterns where before or/and after mask add buffer is reshaped
    auto mask = makePattern();
    // Optional reshape befor adding mask
    auto qk_opt_scaled_pre_mask_shape = any_input();
    auto qk_opt_scaled_pre_mask_opt_reshaped =
        optional<ov::op::v1::Reshape>({qk_opt_scaled, qk_opt_scaled_pre_mask_shape});
    // Optional mask add
    auto qk_opt_scaled_opt_mask_added = optional<ov::op::v1::Add>({qk_opt_scaled_pre_mask_opt_reshaped, mask});
    // Optional reshape after adding mask
    auto qk_post_mask_shape = any_input();
    auto qk_post_mask_opt_reshaped = optional<ov::op::v1::Reshape>({qk_opt_scaled_opt_mask_added, qk_post_mask_shape});

    auto softmax = makePattern<ov::op::v8::Softmax>({qk_post_mask_opt_reshaped}, {{"axis", "-1"}});
    auto softmax_shape = any_input();
    auto softmax_opt_reshaped = optional<ov::op::v1::Reshape>({softmax, softmax_shape});

    auto qkv_base =
        makePattern<ov::op::v0::MatMul>({softmax_opt_reshaped, v}, {{"transpose_a", false}, {"transpose_b", false}});
    auto qkv_shape = any_input();
    auto qkv = optional<ov::op::v1::Reshape>({qkv_base, qkv_shape});

    auto valid_qk_shapes = [](const std::shared_ptr<ov::op::v0::MatMul>& qk_matmul) {
        auto q_pshape = qk_matmul->get_input_partial_shape(0);
        auto k_pshape = qk_matmul->get_input_partial_shape(1);

        // set size idxes to be counted from the end
        const int64_t q_head_size_idx = -1;
        const int64_t k_head_size_idx = qk_matmul->get_transpose_b() ? -1 : -2;

        return q_pshape.size() == k_pshape.size() &&
               (q_pshape.size() == 2 || q_pshape.size() == 3 || q_pshape.size() == 4) &&
               q_pshape[q_head_size_idx].is_static() && k_pshape[k_head_size_idx].is_static() &&
               q_pshape[q_head_size_idx].get_length() == k_pshape[k_head_size_idx].get_length();
    };

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        // make sure there is only one scaling
        if (pattern_map.count(k_opt_transposed_scaled) && pattern_map.count(qk_scaled))
            return false;
        // make sure that if inputs are reshaped the output is reshaped back
        bool inputs_reshaped = pattern_map.count(q) && pattern_map.count(k) && pattern_map.count(v);
        bool output_reshaped = pattern_map.count(qkv);
        if ((inputs_reshaped && !output_reshaped) || (!inputs_reshaped && output_reshaped))
            return false;

        auto q_node = pattern_map.at(q_base);
        auto q_node_ps = q_node.get_partial_shape();
        if (q_node_ps[-1].is_dynamic() || q_node_ps[-3].is_dynamic())
            return false;

        auto k_node = pattern_map.at(k_base);
        auto k_node_ps = k_node.get_partial_shape();

        auto v_node = pattern_map.at(v_base);
        auto v_node_ps = v_node.get_partial_shape();
        if (v_node_ps[-1].is_dynamic() || v_node_ps[-3].is_dynamic())
            return false;

        if (q_node_ps[-1] != k_node_ps[-1]) {
            std::shared_ptr<ov::op::v1::Transpose> k_transpose;
            k_transpose = std::make_shared<ov::op::v1::Transpose>(
                k_node,
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1}));
            k_transpose->set_friendly_name(k->get_friendly_name());
            ov::copy_runtime_info(m.get_matched_nodes(), k_transpose);
            k_node = k_transpose;
            k_node_ps = k_node.get_partial_shape();
        }

        if (k_node_ps[-1].is_dynamic() || k_node_ps[-3].is_dynamic())
            return false;

        // get all important dims from shapes:
        auto N = q_node_ps[0];   // batch size - can be dynamic
        auto H = k_node_ps[-3];  // number of heads of key and value
        auto S = k_node_ps[-2];  // source sequence length - can be dynamic
        auto E = q_node_ps[-1];  // embedding dimension of query and key

        auto T = q_node.get_element_type();
        // make sure that all inputs to SDPA (query, key and value) have the same batch
        if ((k_node_ps[0] != N) || (v_node_ps[0] != N))
            return false;

        // make sure that number of heads of value is the same as for key
        if (v_node_ps[-3] != H)
            return false;

        // make sure that source sequence length of value is the same as for key
        if (v_node_ps[-2] != S)
            return false;

        // make sure that embedding dimension of key is the same as for query
        if (k_node_ps[-1] != E)
            return false;

        if (!valid_qk_shapes(ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(qk).get_node_shared_ptr()))) {
            return false;
        }

        if (pattern_map.at(qk).get_target_inputs().size() > 1 ||
            pattern_map.at(softmax).get_target_inputs().size() > 1) {
            return false;
        }
        if (pattern_map.count(qk_opt_scaled_opt_mask_added) &&
            (pattern_map.at(qk_opt_scaled_opt_mask_added).get_target_inputs().size() > 1 ||
             pattern_map.at(mask).get_partial_shape().size() > 4)) {
            return false;
        }

        ov::Output<ov::Node> scale_node;
        if (pattern_map.count(attn_scale)) {
            scale_node = pattern_map.at(attn_scale);
            auto attn_scale_out_ps = scale_node.get_partial_shape();
            // we need to be able to cast attn_scale layer to Constant layer
            // in order to read actual scale value
            float attn_scale_val = 0;
            if (!ov::op::util::get_constant_value<float>(scale_node.get_node_shared_ptr(), attn_scale_val))
                return false;

            scale_node = ov::op::v0::Constant::create(T, ov::Shape{}, {attn_scale_val});
        } else {
            scale_node = ov::op::v0::Constant::create(T, ov::Shape{}, {1.0});
        }

        Output<ov::Node> mask_input;
        if (pattern_map.count(mask) && pattern_map.count(qk_opt_scaled_opt_mask_added)) {
            ov::Output<ov::Node> qk_out = pattern_map.at(qk_opt_scaled_opt_mask_added);
            // Get shape of the first input
            auto qk_out_ps = qk_out.get_target_inputs().begin()->get_partial_shape();

            mask_input = pattern_map.at(mask);
            auto mask_input_ps = mask_input.get_partial_shape();

            if (!qk_out_ps.rank().is_static() || !mask_input_ps.rank().is_static())
                return false;
            if (qk_out_ps.size() > 4)
                return false;

            std::shared_ptr<ov::op::v0::Unsqueeze> mask_unsqueeze;
            // mask should be broadcastable to qk shape
            if (!ov::PartialShape::broadcast_merge_into(qk_out_ps, mask_input_ps, ov::op::AutoBroadcastType::NUMPY))
                return false;

            if (mask_input_ps.size() < qk_out_ps.size()) {
                size_t rank_diff = qk_out_ps.size() - mask_input_ps.size();
                std::vector<int64_t> axes(rank_diff);
                std::iota(axes.begin(), axes.end(), 0);
                mask_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
                    mask_input,
                    ov::op::v0::Constant::create(ov::element::i64, ov::Shape{rank_diff}, axes));
                mask_unsqueeze->set_friendly_name(mask->get_friendly_name());
                ov::copy_runtime_info(m.get_matched_nodes(), mask_unsqueeze);
                mask_input = mask_unsqueeze;
            }
        } else {
            mask_input = ov::op::v0::Constant::create(T, ov::Shape{}, {0});
        }

        std::shared_ptr<ov::Node> sdpa = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_node,
                                                                                                  k_node,
                                                                                                  v_node,
                                                                                                  mask_input,
                                                                                                  scale_node,
                                                                                                  false);

        sdpa->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa);
        ov::replace_node(m.get_match_root(), sdpa);
        return true;
    };

    auto m = std::make_shared<Matcher>(qkv, "SDPAFusion");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace ov
