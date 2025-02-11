// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SoftPlusDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief SoftPlusDecomposition transformation replaces SoftPlus op to
 * group of operations: log(exp(x) + 1).
 */
class ov::pass::SoftPlusDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SoftPlusDecomposition");
    SoftPlusDecomposition();
};
