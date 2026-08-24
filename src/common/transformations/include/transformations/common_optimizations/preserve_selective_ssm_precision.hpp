// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/// \brief Prevents generic precision conversion from changing SelectiveSSM operations and their inputs.
class TRANSFORMATIONS_API PreserveSelectiveSSMPrecision final : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("PreserveSelectiveSSMPrecision");

    PreserveSelectiveSSMPrecision();
};

}  // namespace ov::pass
