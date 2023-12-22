// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/graph_rewrite.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SoftSignDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ie_transformation_common_api
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
    OPENVINO_RTTI("SoftSignDecomposition", "0");
    SoftSignDecomposition();
};
