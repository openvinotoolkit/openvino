// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     FuseClampAndFakeQuantize detects Clamp -> FakeQuantize patterns for non-binary
 *     FakeQuantize operations, removes the explicit Clamp nodes from the ov::Model,
 *     stores the effective Clamp interval in FakeQuantize runtime info,
 *     and rewires FakeQuantize directly to the source before Clamp.
 *
 * Supported patterns:
 *     1. Clamp -> FakeQuantize, where FakeQuantize levels > 2
 *     2. Chains of consecutive Clamp nodes before FakeQuantize
 *
 * Before:
 *
 * +-----------+
 * |   Input   |
 * +-----+-----+
 *       |
 * +-----v-----+
 * |   Clamp   |
 * +-----+-----+
 *       |
 * +-----v-----+
 * | [Clamp]*  |
 * +-----+-----+
 *       |
 * +-----v-------------+
 * |   FakeQuantize    |
 * +-----+-------------+
 *       |
 * +-----v-----+
 * |  Result   |
 * +-----------+
 *
 * After:
 *
 * +-----------+
 * |   Input   |
 * +-----+-----+
 *       |
 * +-----v----------------------------------+
 * | FakeQuantize + rt_info(ClampBounds)    |
 * +-----+----------------------------------+
 *       |
 * +-----v-----+
 * |  Result   |
 * +-----------+
 *
 */

namespace ov::intel_cpu {

class FuseClampAndFakeQuantize : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseClampAndFakeQuantize");
    FuseClampAndFakeQuantize();
};

}  // namespace ov::intel_cpu