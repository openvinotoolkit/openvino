// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/gn_decomposition.hpp"

#include "openvino/op/group_normalization.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/snippets_isa.hpp"
#include "openvino/core/rt_info.hpp"

namespace ov {
namespace snippets {
namespace pass {
using namespace lowered;

// groupNorm -> reshape + mvn + reshape + mul + add,
// where mvn = (x - mean) / Sqrt(ReduceMean((x - mean) ^ 2) + eps),
// where mean = ReduceMean(x, axes)
GNDecomposition::GNDecomposition() {
    MATCHER_SCOPE(GNDecomposition);
    auto group_norm_pattern = ov::pass::pattern::wrap_type<ov::op::v12::GroupNormalization>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::pass::GNDecomposition")
        auto group_norm_node = ov::as_type_ptr<ov::op::v12::GroupNormalization>(m.get_match_root());
        OPENVINO_ASSERT(!group_norm_node->is_dynamic(), "GroupNormalization decomposition in snippets only support static node.");

        const auto data = group_norm_node->input_value(0);
        const auto scale = group_norm_node->input_value(1);
        const auto bias = group_norm_node->input_value(2);

        const auto num_groups = static_cast<size_t>(group_norm_node->get_num_groups());
        const float eps = static_cast<float>(group_norm_node->get_epsilon());

        ////////////collapse to reduce lastDim to avoid nested loop overhead(e.g. reduce tails in inner loop)///////////
        // reshape [N, C, spatial] to [N, group, 1, (C / group) * spatial]
        const auto orig_shape = group_norm_node->get_input_partial_shape(0).to_shape();
        size_t orig_rank = orig_shape.size();
        OPENVINO_ASSERT(orig_rank >= 2, "First input rank for group normalization op should be greater than 1");
        size_t group_rank = 4;
        size_t c_in_group = orig_shape[1] / num_groups;
        size_t spatial_dim = 1;
        for (size_t i = 2; i < orig_rank; ++i) {
            spatial_dim = spatial_dim * orig_shape[i];
        }
        ov::Shape group_shape = {orig_shape[0], num_groups, 1ul, c_in_group * spatial_dim};
        std::shared_ptr<ov::Node> reshaped_node_orig = std::make_shared<ov::snippets::op::Reshape>(data, group_shape);

        std::shared_ptr<ov::Node> reshaped_node1 = reshaped_node_orig;
        if (data.get_element_type() != element::f32) {
            reshaped_node1 = std::make_shared<ov::snippets::op::ConvertSaturation>(reshaped_node_orig, element::f32);
        }

        const auto reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(reshaped_node1, group_rank - 1);
        op::ReduceBase::compute_and_set_reduce_subtensors(reduce_sum);

        // reduceMean
        float group_size_inv = 1.0f / static_cast<float>(group_shape[3]);
        const auto group_size_inv_node = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{group_size_inv});
        const auto reduce_mean = std::make_shared<ov::op::v1::Multiply>(reduce_sum, group_size_inv_node);

        // x - mean
        std::shared_ptr<ov::Node> reshaped_node2 = reshaped_node_orig;
        if (data.get_element_type() != element::f32) {
            reshaped_node2 = std::make_shared<ov::snippets::op::ConvertSaturation>(reshaped_node_orig, element::f32);
        }
        auto sub_mean = std::make_shared<ov::op::v1::Subtract>(reshaped_node2, reduce_mean);
        // (x - mean) ^ 2
        auto sqr_const = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1}, std::vector<float>{2});
        auto sqr = std::make_shared<ov::op::v1::Power>(sub_mean, sqr_const);
        // reduceSum((x - mean) ^ 2)
        auto sqr_reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(sqr, group_rank - 1);
        op::ReduceBase::compute_and_set_reduce_subtensors(sqr_reduce_sum);
        // reduceMean((x - mean) ^ 2)
        const auto group_size_inv_node_aux = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{}, std::vector<float>{group_size_inv});
        auto sqr_mean = std::make_shared<ov::op::v1::Multiply>(sqr_reduce_sum, group_size_inv_node_aux);
        // reduceMean((x - mean) ^ 2) + eps
        auto eps_node = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1}, std::vector<float>{eps});
        auto eps_add = std::make_shared<ov::op::v1::Add>(sqr_mean, eps_node);  // fma to this add and parent multiply
        // variance = sqrt( reducemean( (x - mean) ^ 2 ) + eps )
        auto variance = std::make_shared<ov::op::v0::Sqrt>(eps_add);
        // divide variance
        const auto variance_inv = std::make_shared<ov::snippets::op::PowerStatic>(variance, -1.f);
        auto mvn = std::make_shared<ov::op::v1::Multiply>(sub_mean, variance_inv);

        // reshape mvn from [N, group, 1, (C / group) * spatial] to [N, group, C / group, spatial]
        ov::Shape group_channel_shape = {orig_shape[0], num_groups, c_in_group, spatial_dim};
        const auto mvn_reshaped = std::make_shared<ov::snippets::op::Reshape>(mvn, group_channel_shape);

        // reshape scale and bias to [1, group, C / group, 1]
        ov::Shape scale_bias_shape = {1ul, num_groups, c_in_group, 1ul};
        std::shared_ptr<ov::Node> reshape_scale = std::make_shared<ov::snippets::op::Reshape>(scale, scale_bias_shape);
        if (scale.get_element_type() != element::f32) {
            reshape_scale = std::make_shared<ov::snippets::op::ConvertSaturation>(reshape_scale, element::f32);
        }
        std::shared_ptr<ov::Node> reshape_bias = std::make_shared<ov::snippets::op::Reshape>(bias, scale_bias_shape);
        if (bias.get_element_type() != element::f32) {
            reshape_bias = std::make_shared<ov::snippets::op::ConvertSaturation>(reshape_bias, element::f32);
        }

        // scaled mvn_reshape[2,5,2,64] reshape_scale[1,5,2,1] -> scaled_node[2,5,2,64]
        auto scaled_node = std::make_shared<ov::op::v1::Multiply>(mvn_reshaped, reshape_scale);
        auto biased_node = std::make_shared<ov::op::v1::Add>(scaled_node, reshape_bias);

        auto result_prec = group_norm_node->get_output_element_type(0);
        std::shared_ptr<ov::Node> biased_node_convert = biased_node;
        if (result_prec != element::f32) {
            biased_node_convert = std::make_shared<ov::snippets::op::ConvertSaturation>(biased_node, result_prec);
        }

        // reshape_back [N, group, C / group, spatial] to [N, C, spatial]
        const auto reshape_back_node = std::make_shared<ov::snippets::op::Reshape>(biased_node_convert, orig_shape);

        return ov::replace_node_update_name(group_norm_node, reshape_back_node);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(group_norm_pattern, matcher_name);
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ov