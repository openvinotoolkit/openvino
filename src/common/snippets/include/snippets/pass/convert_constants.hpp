// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::snippets::pass {

/**
 * @interface ConvertConstantsToScalars
 * @brief Replace only constants which are should be represented as scalars during code generation.
 *        Only single-value (0D) constants are currently supported.
 * @ingroup snippets
 */
class ConvertConstantsToScalars : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("snippets::pass::ConvertConstantsToScalars");
    ConvertConstantsToScalars();
};

}  // namespace ov::snippets::pass
