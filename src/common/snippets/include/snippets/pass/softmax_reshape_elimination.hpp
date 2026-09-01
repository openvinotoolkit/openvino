// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/reshape.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::snippets::pass {

/**
 * @interface SoftmaxReshapeElimination
 * @brief The pass removes Reshape operations around Softmax if possible
 * @ingroup snippets
 */
class SoftmaxReshapeElimination : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::SoftmaxReshapeElimination");
    SoftmaxReshapeElimination();

    static bool eliminate(const std::shared_ptr<ov::op::v1::Reshape>& reshape0,
                          const std::shared_ptr<ov::Node>& softmax,
                          const std::shared_ptr<ov::op::v1::Reshape>& reshape1);
};

}  // namespace ov::snippets::pass
