// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API EliminateGratuitousSliceCascade;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Folds the gratuitous
 *        `Select(Constant<all-false>, ConvertLike(ShapeOf(X), _), Add(Constant, Constant))`
 *        cascade emitted by the TF/TFLite Slice translator (translate_slice_op) into a single
 *        Constant carrying the literal `start + size` values. The else-branch may already be
 *        a folded Constant; in that case the matcher just replaces Select with it.
 *
 * Why this exists: pipelines that consume the model without a follow-up ConstantFolding
 * pass (notably NPUW's pre-partition stage) otherwise carry the dynamic Select node into
 * partitioning, where `to_shape()` on the Slice output throws.
 */
class ov::pass::EliminateGratuitousSliceCascade : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("EliminateGratuitousSliceCascade");
    EliminateGratuitousSliceCascade();
};
