// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/group_normalization_decomposition.hpp"

#include "openvino/op/group_normalization.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "snippets/itt.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/snippets_isa.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/core/rt_info.hpp"

namespace ov {
namespace snippets {
namespace pass {
using namespace lowered;

// groupNorm -> reshape + mvn + reshape + mul + add
// mvn -> (x - ReduceMean(x, axes)) / Sqrt(ReduceMean((x - ReduceMean(x, axes)) ^ 2) + eps)
GroupNormalizationDecomposition::GroupNormalizationDecomposition() {
    MATCHER_SCOPE(GroupNormalizationDecomposition);
    auto group_norm_pattern = ov::pass::pattern::wrap_type<ov::op::v12::GroupNormalization>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::pass::GroupNormalizationDecomposition")
        auto group_norm_node = ov::as_type_ptr<ov::op::v12::GroupNormalization>(m.get_match_root());

        const auto data = group_norm_node->input_value(0);
        const auto scale = group_norm_node->input_value(1);
        const auto bias = group_norm_node->input_value(2);

        const auto num_groups = static_cast<size_t>(group_norm_node->get_num_groups());
        const auto eps = ov::op::util::cast_eps_to_float(group_norm_node->get_epsilon());

        // reshape [N, C, spatial] to [N, group, C / group, spatial]
        const auto orig_shape = group_norm_node->get_input_shape(0);
        size_t orig_rank = orig_shape.size();
        ov::Shape group_shape(orig_rank + 1);
        group_shape[0] = orig_shape[0];
        group_shape[1] = num_groups;
        group_shape[2] = orig_shape[1] / num_groups;
        for (size_t i = 3; i < orig_rank + 1; ++i) {
            group_shape[i] = orig_shape[i - 1];
        }
        auto group_shape_node = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{group_shape.size()}, group_shape);
        const auto reshaped_node = std::make_shared<ov::op::v1::Reshape>(data, group_shape_node, true);

        // reduceSum on dimension [C / group, spatial]
        int64_t axis_start = 2;
        std::vector<int64_t> axis(group_shape.size() - axis_start);
        std::iota(axis.begin(), axis.end(), axis_start); // axis:[2, 3, 4...]
        auto axis_node = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{axis.size()}, axis);
        // todo: snippets op ReduceSum to have emitter to generate
        const auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(reshaped_node, axis_node, true);

        // reduceMean
        int64_t group_size = std::accumulate(group_shape.begin() + axis_start, group_shape.end(), 1, std::multiplies<int64_t>());
        auto group_size_node = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{}, std::vector<int64_t>{group_size});
        const auto group_size_inv = std::make_shared<ov::snippets::op::PowerStatic>(group_size_node, -1.f);
        const auto reduce_mean = std::make_shared<ov::op::v1::Multiply>(reduce_sum, group_size_inv);

        // x - mean
        auto mean_norm = std::make_shared<ov::op::v1::Subtract>(reshaped_node, reduce_mean);
        // (x - mean) ^ 2
        auto sqr_const = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{2});
        auto sqr = std::make_shared<ov::op::v1::Power>(mean_norm, sqr_const);
        // reduceSum((x - mean) ^ 2)
        // todo: snippets op ReduceSum to have emitter to generate
        auto mean_sum_variance = std::make_shared<ov::op::v1::ReduceSum>(sqr, axis_node, true);
        // reduceMean((x - mean) ^ 2)
        auto reduce_mean_variance = std::make_shared<ov::op::v1::Multiply>(mean_sum_variance, group_size_inv);
        // reduceMean((x - mean) ^ 2) + eps
        auto eps_node = std::make_shared<ov::op::v0::Constant>(element::f32, Shape{1}, std::vector<float>{eps});
        auto eps_add = std::make_shared<ov::op::v1::Add>(reduce_mean_variance, eps_node);
        // variance = sqrt( reducemean( (x - mean) ^ 2 ) + eps )
        auto variance = std::make_shared<ov::op::v0::Sqrt>(eps_add);

        // ( (x - mean) / variance) * scale + bias
        // reshape scale and bias
        std::vector<size_t> c_shape(group_shape.size(), 1);
        c_shape[1] = group_shape[1];
        c_shape[2] = group_shape[2];
        auto c_reshape = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{c_shape.size()}, c_shape);
        const auto reshape_scale = std::make_shared<ov::op::v1::Reshape>(scale, c_reshape, true);
        const auto reshape_bias = std::make_shared<ov::op::v1::Reshape>(bias, c_reshape, true);

        // e.g, orig[2,10,8,8], reshaped[2,5,2,8,8],
        // variance_inv[2,5,1,1,1] reshape_scale[1,5,2,1,1] -> result[2,5,2,1,1]
        const auto variance_inv = std::make_shared<ov::snippets::op::PowerStatic>(variance, -1.f);
        auto scaled_variance_inv = std::make_shared<ov::op::v1::Multiply>(variance_inv, reshape_scale);
        // this enable MulAddToFMA afterwards
        // mean_norm[2,5,2,8,8] scaled_variance_inv[2,5,2,1,1] -> result[2,5,2,8,8]
        auto scaled_node = std::make_shared<ov::op::v1::Multiply>(mean_norm, scaled_variance_inv);
        // scaled_node[2,5,2,8,8] scaled_node[1,5,2,1,1] -> result[2,5,2,8,8]
        auto biased_node = std::make_shared<ov::op::v1::Add>(scaled_node, scaled_node);

        // reshape_back [N, group, C / group, spatial] to [N, C, spatial]
        auto orig_shape_node = std::make_shared<ov::op::v0::Constant>(element::i64, Shape{orig_shape.size()}, orig_shape);
        const auto reshape_back_node = std::make_shared<ov::op::v1::Reshape>(biased_node, orig_shape_node, true);

        ov::replace_node_update_name(group_norm_node, biased_node);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(group_norm_pattern, matcher_name);
    register_matcher(m, callback);
}

}  // namespace pass
}  // namespace snippets
}  // namespace ov
