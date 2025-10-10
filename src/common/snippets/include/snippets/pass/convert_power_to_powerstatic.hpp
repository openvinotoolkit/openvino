// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "snippets/snippets_visibility.hpp"

namespace ov::snippets::pass {

/**
 * @interface ConvertConstantsToScalars
 * @brief Replace Power with a scalar input with snippets::op::PowerStatic for generation of a more optimal code.
 * @ingroup snippets
 */
class SNIPPETS_API ConvertPowerToPowerStatic : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::ConvertPowerToPowerStatic");
    ConvertPowerToPowerStatic();
};

}  // namespace ov::snippets::pass
