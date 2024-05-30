// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_group_normalization.hpp"
#include <snippets/op/subgraph.hpp>

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> GroupNormalizationFunction::initOriginal() const {
    auto data = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto scale = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto shift = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    const auto groupNormalization = std::make_shared<ov::op::v12::GroupNormalization>(data, scale, shift, num_groups, epsilon);
    return std::make_shared<ov::Model>(NodeVector{groupNormalization}, ParameterVector{data, scale, shift});
}

std::shared_ptr<ov::Model> GroupNormalizationFunction::initReference() const {
    auto data = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto scale = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto shift = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    auto data_ = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto scale_ = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto shift_ = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);
    const auto groupNormalization = std::make_shared<ov::op::v12::GroupNormalization>(data_, scale_, shift_, num_groups, epsilon);

    auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{data, scale, shift},
            std::make_shared<ov::Model>(NodeVector{groupNormalization}, ParameterVector{data_, scale_, shift_}));

    return std::make_shared<ov::Model>(NodeVector{subgraph}, ParameterVector{data, scale, shift});
}

std::shared_ptr<ov::Model> GroupNormalizationFunction::initLowered() const {
    auto data = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto scale = std::make_shared<op::v0::Parameter>(precision, input_shapes[1]);
    auto bias = std::make_shared<op::v0::Parameter>(precision, input_shapes[2]);

    // reshape [N, C, spatial] to [N, group, 1, (C / group) * spatial]
    const auto orig_shape = input_shapes[0].to_shape();
    size_t orig_rank = orig_shape.size();
    size_t group_rank = 4;
    size_t c_in_group = orig_shape[1] / num_groups;
    size_t spatial_dim = 1;
    for (size_t i = 2; i < orig_rank; ++i) {
        spatial_dim = spatial_dim * orig_shape[i];
    }
    ov::Shape group_shape = {orig_shape[0], num_groups, 1ul, c_in_group * spatial_dim};
    std::shared_ptr<ov::Node> reshaped_node_orig = std::make_shared<ov::snippets::op::Reshape>(data, group_shape);
    const auto reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(reshaped_node_orig, group_rank - 1);

    // reduceMean
    float group_size_inv = 1.0f / static_cast<float>(group_shape[3]);
    // scalar const -> scalar in data_flow_optimization.
    const auto group_size_inv_node = std::make_shared<ov::snippets::op::Scalar>(element::f32, Shape{1}, group_size_inv);
    const auto reduce_mean = std::make_shared<ov::op::v1::Multiply>(reduce_sum, group_size_inv_node);

    // x - mean
    std::shared_ptr<ov::Node> reshaped_node2 = reshaped_node_orig;
    auto sub_mean = std::make_shared<ov::op::v1::Subtract>(reshaped_node2, reduce_mean);
    // (x - mean) ^ 2
    // power -> poweStatic in data_flow_optimization
    auto sqr = std::make_shared<ov::snippets::op::PowerStatic>(sub_mean, 2.0f);
    // reduceSum((x - mean) ^ 2)
    auto sqr_reduce_sum = std::make_shared<ov::snippets::op::ReduceSum>(sqr, group_rank - 1);
    // reduceMean((x - mean) ^ 2)
    const auto group_size_inv_node_aux = std::make_shared<ov::snippets::op::Scalar>(element::f32, Shape{1}, group_size_inv);
    auto sqr_mean = std::make_shared<ov::op::v1::Multiply>(sqr_reduce_sum, group_size_inv_node_aux);
    // reduceMean((x - mean) ^ 2) + eps
    auto eps_node = std::make_shared<ov::snippets::op::Scalar>(element::f32, Shape{1}, epsilon);
    auto eps_add = std::make_shared<ov::op::v1::Add>(sqr_mean, eps_node);
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
    std::shared_ptr<ov::Node> reshape_bias = std::make_shared<ov::snippets::op::Reshape>(bias, scale_bias_shape);

    auto scaled_node = std::make_shared<ov::op::v1::Multiply>(mvn_reshaped, reshape_scale);
    auto biased_node = std::make_shared<ov::op::v1::Add>(scaled_node, reshape_bias);

    // reshape_back [N, group, C / group, spatial] to [N, C, spatial]
    const auto reshape_back_node = std::make_shared<ov::snippets::op::Reshape>(biased_node, orig_shape);

    return std::make_shared<ov::Model>(NodeVector{reshape_back_node}, ParameterVector{data, scale, bias});
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
