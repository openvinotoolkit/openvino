// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"

namespace ov {
namespace snippets {
namespace pass {

/**
 * @interface ConvertConstantsToScalars
 * @brief Replace only constants which are should be represented as scalars during code generation.
 *        Only single-value (0D) constants are currently supported.
 * @ingroup snippets
 */
class ConvertConstantsToScalars: public ov::pass::MatcherPass {
public:
    ConvertConstantsToScalars();
};

} // namespace pass
} // namespace snippets
} // namespace ov
