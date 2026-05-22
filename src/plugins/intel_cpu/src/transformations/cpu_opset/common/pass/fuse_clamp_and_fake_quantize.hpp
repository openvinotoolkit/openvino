// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     FuseClampAndFakeQuantize detects redundant Clamp -> FakeQuantize patterns,
 *     removes Clamp nodes whose interval fully covers the FakeQuantize input interval,
 *     and rewires FakeQuantize directly to the source before Clamp.
 *
 * Supported patterns:
 *     1. Clamp -> FakeQuantize, where Clamp range is wider than or equal to the
 *        FakeQuantize input interval
 *     2. Chains of consecutive redundant Clamp nodes before FakeQuantize
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
 * +-----v-------------+
 * |   FakeQuantize    |
 * +-----+-------------+
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