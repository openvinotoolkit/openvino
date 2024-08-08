// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"

#include "snippets/op/brgemm.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface FlashAttentionTransformation
 * @brief Transform MHA to operations that could perform computation on partial tensor. This materialize split K ana V matrix
 *     and enable MM0 + ... + softmax + ... + MM1 vertical fusion on splitted K/V subtensor.
 * @ingroup snippets
 */
class FlashAttentionTransformation: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FlashAttentionTransformation", "0");
    FlashAttentionTransformation();
};

} // namespace pass
} // namespace snippets
} // namespace ov
