// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/sdpa_fusion.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/gen_pattern.hpp"

namespace ov {
namespace pass {

SDPAFusion::SDPAFusion() {
    using namespace ov::pass::pattern;
    using namespace ov::gen_pattern;

    auto q_base = makePattern(ov::Rank(4));
    auto q_proj_shape_m = ov::pass::pattern::any_input();
    auto q_reshaped= ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({q_base, q_proj_shape_m});
    auto q = q_reshaped;

    auto q_slice_start= ov::pass::pattern::any_input();
    auto q_slice_stop = ov::pass::pattern::any_input();
    auto q_slice_step = ov::pass::pattern::any_input();
    auto q_slice_axes = ov::pass::pattern::any_input();
    auto q_reshaped_sliced_m = 
    ov::pass::pattern::wrap_type<ov::op::v8::Slice>({q_reshaped, q_slice_start, q_slice_stop, q_slice_step, q_slice_axes});

    auto q_proj_shape2_m = ov::pass::pattern::any_input();
    auto q_proj_reshaped_sliced_reshaped_m = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({q_reshaped_sliced_m, q_proj_shape2_m});
    auto q_proj_reshaped_opt_sliced_reshaped_m = q_proj_reshaped_sliced_reshaped_m | q_reshaped;



    auto k_base = makePattern(ov::Rank(4));
    auto k_proj_shape_m = ov::pass::pattern::any_input();
    auto k_reshaped = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({k_base, k_proj_shape_m});
    auto k = k_reshaped;

    auto k_slice_start= ov::pass::pattern::any_input();
    auto k_slice_stop = ov::pass::pattern::any_input();
    auto k_slice_step = ov::pass::pattern::any_input();
    auto k_slice_axes = ov::pass::pattern::any_input();
    auto k_reshaped_sliced_m = 
    ov::pass::pattern::wrap_type<ov::op::v8::Slice>({k_reshaped, k_slice_start, k_slice_stop, k_slice_step, k_slice_axes});

    auto k_proj_shape2_m = ov::pass::pattern::any_input();
    auto k_proj_reshaped_sliced_reshaped_m = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({k_reshaped_sliced_m, k_proj_shape2_m});
    auto k_proj_reshaped_opt_sliced_reshaped_m = k_proj_reshaped_sliced_reshaped_m | k_reshaped;
 

    auto attn_scale_m = ov::pass::pattern::any_input();
    auto k_scaled = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({k_proj_reshaped_opt_sliced_reshaped_m, attn_scale_m});
    auto k_opt_scaled = k_scaled | k_proj_reshaped_opt_sliced_reshaped_m;
    auto k_proj_trans_dims_m = ov::pass::pattern::any_input();
    auto k_proj_transposed_m =
        ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({k_proj_reshaped_opt_sliced_reshaped_m, k_proj_trans_dims_m});
    auto k_proj_transposed_scaled_m =
        ov::pass::pattern::wrap_type<ov::op::v1::Multiply>(ov::OutputVector{k_proj_transposed_m, attn_scale_m});
    auto k_proj_transposed_opt_scaled_m = k_proj_transposed_m | k_proj_transposed_scaled_m;
    auto k_proj_opt_transposed_opt_scaled_m = k_opt_scaled | k_proj_transposed_opt_scaled_m;
    
    auto v_base = makePattern(ov::Rank(4));
    auto v_proj_shape_m = ov::pass::pattern::any_input();
    auto v_reshaped = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({v_base, v_proj_shape_m});
    auto v = v_reshaped;

    auto v_slice_start= ov::pass::pattern::any_input();
    auto v_slice_stop = ov::pass::pattern::any_input();
    auto v_slice_step = ov::pass::pattern::any_input();
    auto v_slice_axes = ov::pass::pattern::any_input();
    auto v_reshaped_sliced_m = 
    ov::pass::pattern::wrap_type<ov::op::v8::Slice>({v_reshaped, v_slice_start, v_slice_stop, v_slice_step, v_slice_axes});

    auto v_proj_shape2_m = ov::pass::pattern::any_input();
    auto v_proj_reshaped_sliced_reshaped_m =
        ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({v_reshaped_sliced_m, v_proj_shape2_m});
    auto v_proj_reshaped_opt_sliced_reshaped_m = v_proj_reshaped_sliced_reshaped_m | v_reshaped;

    auto v_proj_trans_dims_m = ov::pass::pattern::any_input();
    auto v_proj_transposed_m = ov::pass::pattern::wrap_type<ov::op::v1::Transpose>(
        {v_proj_reshaped_opt_sliced_reshaped_m, v_proj_trans_dims_m});
    auto v_proj_transposed_opt_m = v_proj_reshaped_opt_sliced_reshaped_m | v_proj_transposed_m;


    auto mask = makePattern();

    // auto k_transpose_order = pattern::wrap_type<ov::op::v0::Constant>([](const Output<Node>& node) {
    //     auto axis_order = ov::as_type_ptr<ov::op::v0::Constant>(node.get_node_shared_ptr())->cast_vector<int64_t>();
    //     return axis_order == std::vector<int64_t>{0, 2, 1};
    // });

    // auto k_t = pattern::wrap_type<ov::op::v1::Transpose>({k_opt_scaled, k_transpose_order});
    auto qk_nn = makePattern<ov::op::v0::MatMul>({q_proj_reshaped_opt_sliced_reshaped_m, k_proj_opt_transposed_opt_scaled_m});
    // auto qk_nt = makePattern<ov::op::v0::MatMul>({q, k_opt_scaled}, {{"transpose_a", false}, {"transpose_b", true}});
    auto qk = qk_nn;

    auto unsqueeze_axis = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto qk_unsqueeze = ov::pass::pattern::wrap_type<ov::op::v0::Unsqueeze>({qk, unsqueeze_axis});
    auto qk_opt_unsqueeze = qk_unsqueeze | qk;
    auto qk_opt_unsqueeze_concat =
        ov::pass::pattern::wrap_type<ov::op::v0::Concat>(ov::OutputVector{qk_opt_unsqueeze}, 0);
    auto qk_opt_unsqueeze_opt_concat = qk_opt_unsqueeze_concat | qk_opt_unsqueeze;
    

    auto qk_scaled = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({qk_opt_unsqueeze_opt_concat, attn_scale_m});
    auto qk_opt_scaled = qk_scaled | qk_opt_unsqueeze_opt_concat;

    // auto attn_bias_m = ov::pass::pattern::any_input();
    
    auto qk_opt_scaled_pre_bias_shape = ov::pass::pattern::any_input();
    auto qk_opt_scaled_pre_bias_reshaped =
        ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({qk_opt_scaled, qk_opt_scaled_pre_bias_shape});
    auto qk_opt_scaled_pre_bias_opt_reshaped = qk_opt_scaled_pre_bias_reshaped | qk_opt_scaled;
   
    auto qk_opt_scaled_biased =
        ov::pass::pattern::wrap_type<ov::op::v1::Add>({qk_opt_scaled_pre_bias_opt_reshaped, mask});
    

    auto qk_opt_scaled_opt_biased = qk_opt_scaled_biased | qk_opt_scaled_pre_bias_opt_reshaped;
    auto qk_post_bias_shape_m = ov::pass::pattern::any_input();
    auto qk_post_bias_reshaped_m =
        ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({qk_opt_scaled_opt_biased, qk_post_bias_shape_m});
    auto qk_post_bias_opt_reshaped = qk_post_bias_reshaped_m | qk_opt_scaled_opt_biased;

    // auto optional_add_mask = optional<ov::op::v1::Add>({qk, mask});
    auto softmax = makePattern<ov::op::v8::Softmax>({qk_post_bias_opt_reshaped}, {{"axis", "-1"}});
    auto softmax_shape_m = ov::pass::pattern::any_input();
    auto softmax_reshaped_m = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({softmax, softmax_shape_m});

    auto softmax_opt_reshaped_m = softmax_reshaped_m | softmax;

    auto softmax_slice_start= ov::pass::pattern::any_input();
    auto softmax_slice_stop = ov::pass::pattern::any_input();
    auto softmax_slice_step = ov::pass::pattern::any_input();
    auto softmax_slice_axes = ov::pass::pattern::any_input();
    auto softmax_reshaped_sliced_m = 
    ov::pass::pattern::wrap_type<ov::op::v8::Slice>({softmax_opt_reshaped_m, softmax_slice_start, softmax_slice_stop, softmax_slice_step, softmax_slice_axes});


    auto softmax_shape2_m = ov::pass::pattern::any_input();
    auto softmax_reshaped_sliced_reshaped_m = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({softmax_reshaped_sliced_m, softmax_shape2_m});
    auto softmax_reshaped_opt_sliced_reshaped_m = softmax_reshaped_sliced_reshaped_m | softmax_opt_reshaped_m;


    auto qkv_base = makePattern<ov::op::v0::MatMul>({softmax_reshaped_opt_sliced_reshaped_m, v_proj_transposed_opt_m}, {{"transpose_a", false}, {"transpose_b", false}});
    auto qkv_shape = ov::pass::pattern::any_input();
    auto qkv_reshaped = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({qkv_base, qkv_shape});
    auto qkv = qkv_reshaped | qkv_base;

    auto valid_qk_shapes = [](const std::shared_ptr<ov::op::v0::MatMul>& qk_matmul) {
        auto q_pshape = qk_matmul->get_input_partial_shape(0);
        auto k_pshape = qk_matmul->get_input_partial_shape(1);

        //set size idxes to be counted from the end
        const int64_t q_head_size_idx = -1;
        const int64_t k_head_size_idx = qk_matmul->get_transpose_b() ? -1 : -2;

        return q_pshape.size() == k_pshape.size() && 
               (q_pshape.size() == 2 || q_pshape.size() == 3 || q_pshape.size() == 4) &&
               q_pshape[q_head_size_idx].is_static() &&
               k_pshape[k_head_size_idx].is_static() &&
               q_pshape[q_head_size_idx].get_length() == k_pshape[k_head_size_idx].get_length();
    };

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto q_node = pattern_map.at(q_base);
        auto q_node_ps = q_node.get_partial_shape();

        auto k_node = pattern_map.at(k_base);
        auto k_node_ps = k_node.get_partial_shape();

        auto v_node = pattern_map.at(v_base);
        auto v_node_ps = v_node.get_partial_shape();

        if (v_node_ps[-2] != k_node_ps[-2]) {
                auto constant = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
                k_node = std::make_shared<ov::op::v1::Transpose>(k_node, constant);
                k_node_ps = k_node.get_partial_shape();
            }

        if (!valid_qk_shapes(ov::as_type_ptr<ov::op::v0::MatMul>(pattern_map.at(qk).get_node_shared_ptr()))) {
            return false;
        }

        if (pattern_map.at(qk).get_target_inputs().size() > 1 ||
            pattern_map.at(softmax).get_target_inputs().size() > 1) {
            return false;
        }
         if (pattern_map.count(qk_opt_scaled_biased) && (pattern_map.at(qk_opt_scaled_biased).get_target_inputs().size() > 1 ||
                                                     pattern_map.at(mask).get_partial_shape().size() > 4)) {
            return false;
        }

        Output<ov::Node> mask_value;
        Output<ov::Node> mask_input;
        if (pattern_map.find(qk_opt_scaled_biased) != pattern_map.end()) {
            mask_value = pattern_map.at(mask);
        } else {
            mask_value = ov::op::v0::Constant::create(q_node.get_element_type(), ov::Shape{}, std::vector<float>{0});
        }
           

        if (mask_value.get_partial_shape().size() > 4) {
            return false;
        }

        if (mask_value.get_partial_shape().rank() == 0 || mask_value.get_partial_shape().rank() == 4) {
            mask_input = mask_value;
        } else {
            size_t rank_diff = q_node.get_partial_shape().size() - mask_value.get_partial_shape().size();
            std::vector<int64_t> axes(rank_diff);
            std::iota(axes.begin(), axes.end(), 0);
            mask_input = std::make_shared<ov::op::v0::Unsqueeze>(
                mask_value,
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{rank_diff}, axes));
        }

        std::shared_ptr<ov::Node> scale_node =
            ov::op::v0::Constant::create(q_node.get_element_type(), ov::Shape{}, std::vector<float>{1.0f});

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

    auto m = std::make_shared<ov::pass::pattern::Matcher>(qkv, "SDPAFusion");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace ov
