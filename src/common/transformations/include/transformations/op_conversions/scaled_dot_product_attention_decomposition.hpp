// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ScaledDotProductAttentionDecomposition;

}  // namespace pass
}  // namespace ov


class ov::pass::ScaledDotProductAttentionDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ScaledDotProductAttentionDecomposition", "0");
    ScaledDotProductAttentionDecomposition();
};
