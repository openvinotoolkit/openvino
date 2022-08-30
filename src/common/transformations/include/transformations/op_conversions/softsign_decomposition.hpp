// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SoftSignDecomposition;

}  // namespace pass
}  // namespace ngraph

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

class ngraph::pass::SoftSignDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SoftSignDecomposition", "0");
    SoftSignDecomposition();
};
