// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/pass/graph_rewrite.hpp>
#include <transformations_visibility.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API SoftmaxDecomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief SoftmaxDecomposition transformation replaces softmax with following graph:
 *
 *            +---------------+
 *            │               │
 *            │     input     │
 *            │               │
 *            +---------------+
 *                │      │
 *                │      v
 *                │ +-----------+
 *                │ │           │
 *                │ │ ReduceMax │
 *                │ │           │
 *                │ +-----------+
 *                │      │
 *                │      │
 *                v      v
 *            +---------------+
 *            │               │
 *            │      Sub      │
 *            │               │
 *            +---------------+
 *                    |
 *                    |
 *                    v
 *            +---------------+
 *            │               │
 *            │      Exp      │
 *            │               │
 *            +---------------+
 *                │      │
 *                │      v
 *                │ +-----------+
 *                │ │           │
 *                │ │ ReduceSum │
 *                │ │           │
 *                │ +-----------+
 *                │      │
 *                │      │
 *                v      v
 *             +-------------+
 *             |             │
 *             |     Div     │
 *             │             │
 *             +-------------+
 *
 */

class ngraph::pass::SoftmaxDecomposition : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("SoftmaxDecomposition", "0");
    SoftmaxDecomposition();
};
