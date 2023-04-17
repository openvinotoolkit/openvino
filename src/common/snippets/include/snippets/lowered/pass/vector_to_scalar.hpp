// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "transformation.hpp"

namespace ngraph {
namespace snippets {
namespace lowered {
namespace pass {

/**
 * @interface SetScalarCountForLoadStore
 * @brief Set count `1` for Load and Store to represent as ScalarLoad / ScalarStore
 * The pass is used to change element count to loading to "1" to load  or store scalar value
 * Used for tail generation
 * @ingroup snippets
 */

// Note that, BrodacastMove is typically inserted right after the Load. Such cases are typical for
// simple subgraphs where one of the ngraph::op's inputs is broadcasted to match the larger one. However, BroadcastMove
// could also be inserted after the ngraph::op, if the op input don't need broadcasting, but the output does
// (for example, to match the larger output of a child node). In such cases, Loads (and Stores) should be replaced
// with ScalarLoads (ScalarStores) to avoid invalid read in vector Loop. Graph example:
// Parameter_0    Parameter_1        Parameter_2
// [1,2,5,16]      [1,2,5,1]          [1,2,5,1]
//   Load        BroadcastLoad         Load*       Scalar
//          Add                             Subtract
//            \___________     ___________BroadcastMove
//                        \   /
//                       Multiply
//                         Store
//                        Result
// Note: Load* should be replaced with ScalarLoad in this example to avoid invalid read in vector Loop.

class SetScalarCountForLoadStore : public Transformation {
public:
    explicit SetScalarCountForLoadStore();
    OPENVINO_RTTI("SetScalarCountForLoadStore", "Transformation")
    bool run(lowered::LinearIR& linear_ir) override;
};

} // namespace pass
} // namespace lowered
} // namespace snippets
} // namespace ngraph
