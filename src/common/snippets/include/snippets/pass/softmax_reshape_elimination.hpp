// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface SoftmaxReshapeElimination
 * @brief The pass removes Reshape operations around Softmax if possible
 * @ingroup snippets
 */
class SoftmaxReshapeElimination: public ov::pass::MatcherPass {
public:
    SoftmaxReshapeElimination();
};


}  // namespace pass
}  // namespace snippets
}  // namespace ov
