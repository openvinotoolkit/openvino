// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <set>

#include "lpt_visibility.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "quantization_details.hpp"

namespace ov {
namespace pass {
namespace low_precision {

/**
 * @ingroup ov_transformation_common_api
 * @brief FQStrippingTransformation strips FakeQuantize operations with specified levels
 * by replacing them with Clamp operations.
 */
class LP_TRANSFORMATIONS_API FQStrippingTransformation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("FQStrippingTransformation", "0", MatcherPass);
    FQStrippingTransformation(const std::set<size_t>& levels_to_strip, bool replace_with_clamp);
};

} // namespace low_precision
} // namespace pass
} // namespace ov