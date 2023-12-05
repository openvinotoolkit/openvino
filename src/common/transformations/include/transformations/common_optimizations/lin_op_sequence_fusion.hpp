// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API LinOpSequenceFusion;
class TRANSFORMATIONS_API AddMultiplyFusion;
class TRANSFORMATIONS_API AddAddFusion;
class TRANSFORMATIONS_API MultiplyMultiplyFusion;

}  // namespace pass
}  // namespace ov

class ov::pass::AddMultiplyFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("AddMultiplyFusion", "0");
    AddMultiplyFusion();
};

class ov::pass::AddAddFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("AddAddFusion", "0");
    AddAddFusion();
};

class ov::pass::MultiplyMultiplyFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MultiplyMultiplyFusion", "0");
    MultiplyMultiplyFusion();
};

/**
 * @ingroup ie_transformation_common_api
 * @brief LinOpSequenceFusion transformation fuses linear operation sequence.
 */
class ov::pass::LinOpSequenceFusion : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("LinOpSequenceFusion", "0");
    LinOpSequenceFusion() {
        add_matcher<ov::pass::AddMultiplyFusion>();
        add_matcher<ov::pass::AddAddFusion>();
        add_matcher<ov::pass::MultiplyMultiplyFusion>();
    }
};
