// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/// \brief GatedDeltaNetFusion replaces the Loop-based recurrent attention graph
///        emitted by optimum-intel with one GatedDeltaNet internal operation.
///
/// The matched graph has head-first inputs [batch, num_heads, seq_len, dim] and
/// is wrapped as:
///
///   Concat(
///       Reshape(Loop(...).output(0), [-1]),
///       Reshape(Loop(...).output(1), [-1]),
///       axis=0)
///
/// The Loop body slices axis 2 one token at a time and applies the recurrent
/// update with ScatterUpdate-based accumulation of the full attention output.
///
/// The fusion transposes external inputs to the seq-first layout required by
/// GatedDeltaNet, creates the internal op, then transposes and reshapes its
/// outputs back so the original flattened concat interface is preserved.
class TRANSFORMATIONS_API GatedDeltaNetFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("GatedDeltaNetFusion", "0");
    GatedDeltaNetFusion();
};

}  // namespace pass
}  // namespace ov
