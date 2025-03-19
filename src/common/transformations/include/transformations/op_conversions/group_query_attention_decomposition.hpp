// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API GroupQueryAttentionDecomposition;

}  // namespace pass
}  // namespace ov

class ov::pass::GroupQueryAttentionDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GroupQueryAttentionDecomposition");
    GroupQueryAttentionDecomposition();

private:
    ov::OutputVector decompose(std::shared_ptr<ov::op::internal::GroupQueryAttention> node);
    std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<op::v3::ShapeOf>& shape,
                                             const std::vector<int>& dims);
    std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims);
    ov::OutputVector make_split(const ov::Output<ov::Node>& value, int64_t num_splits, int64_t axis);
    std::shared_ptr<ov::Node> rotaryEmbedding(ov::Output<ov::Node> input,
                                              ov::Output<ov::Node> past_seqlen,
                                              std::shared_ptr<ov::Node> seqlen_k,
                                              std::shared_ptr<ov::Node> cos_cache,
                                              std::shared_ptr<ov::Node> sin_cache,
                                              std::shared_ptr<ov::Node> dim_head_size,
                                              bool interleaved);
};
