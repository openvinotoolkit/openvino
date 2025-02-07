// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/group_query_attention.hpp"
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
    ov::OutputVector decompose(std::shared_ptr<ov::op::v15::GroupQueryAttention> node);
};
