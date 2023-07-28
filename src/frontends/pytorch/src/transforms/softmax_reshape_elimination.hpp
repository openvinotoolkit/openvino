// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

/**
 * @interface SoftmaxReshapeElimination
 * @brief The pass removes Reshape operations around Softmax if possible
 * @ingroup snippets
 */
class SoftmaxReshapeElimination : public ov::pass::MatcherPass {
public:
    SoftmaxReshapeElimination();
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov