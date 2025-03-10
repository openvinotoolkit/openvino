// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SoftSignDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
 * @brief SoftSignDecomposition transformation replaces SoftSign with the following graph
 *
 *       Input ---> Abs
 *         |         |
 *         |         |
 *         |         |
 *         |         V
 *         |        Add <--- 1
 *         |         |
 *         |         |
 *         V         |
 *       Divide <----|
 *         |
 *         |
 *         |
 *         V
 *       Output
 */

class ov::pass::SoftSignDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SoftSignDecomposition");
    SoftSignDecomposition();
};
