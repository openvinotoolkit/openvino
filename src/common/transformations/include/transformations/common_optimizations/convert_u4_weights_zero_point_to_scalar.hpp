// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API ConvertU4WeightsZeroPointToScalar;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief Converts U4 weights zero point to scalar if all values are equal
 */
class ov::pass::ConvertU4WeightsZeroPointToScalar : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertU4WeightsZeroPointToScalar");
    ConvertU4WeightsZeroPointToScalar();
};
