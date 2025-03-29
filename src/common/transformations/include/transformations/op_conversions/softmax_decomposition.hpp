// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API SoftmaxDecomposition;

}  // namespace pass
}  // namespace ov

/**
 * @ingroup ov_transformation_common_api
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

class ov::pass::SoftmaxDecomposition : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SoftmaxDecomposition");
    SoftmaxDecomposition();
};
