// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @brief Preserve the shared-expert GatherMatmul activation layout during PA conversion.
 *
 * A [1, tokens, channels] input is reshaped to [1, -1, 0] before PA flattens
 * the model's token inputs. The reshape is initially an identity but keeps the
 * expert axis at one when upstream activations become [tokens, 1, channels].
 * Non-unit or unknown expert axes are left unchanged.
 *
 * Before: [1, tokens, channels] -> GatherMatmul
 * After:  [1, tokens, channels] -> Reshape(1, -1, 0) -> GatherMatmul
 */
class TRANSFORMATIONS_API PreserveGatherMatmulLayout : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PreserveGatherMatmulLayout");
    PreserveGatherMatmulLayout();
};

}  // namespace ov::pass
