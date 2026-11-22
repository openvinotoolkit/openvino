// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

/**
 * @brief Keeps integer ReduceProd in i64 so that products beyond the i32 range stay exact.
 *
 * The CPU plugin lowers every i64 tensor to i32 in ConvertPrecision. For ReduceProd this silently
 * overflows, e.g. PyTorch `torch.prod` of int32 inputs returns an int64 result which is not representable
 * in i32 (see openvinotoolkit/openvino#38719). The Reduce node has an exact i64 reference kernel, so
 * the pass opts the ReduceProd and the Convert that feeds it out of the i64->i32 conversion:
 *
 *      X(any int) -> Convert(i64) -> ReduceProd(i64) -> Result
 *                                                    -> Convert(i64) -> <other consumers>
 *
 * Both nodes are marked with @ref ov::disable_conversion(node, i64, i32). Consumers other than Result
 * are routed through a new, unmarked Convert, which ConvertPrecision lowers to i32 as usual, so the
 * rest of the graph keeps the plugin's native integer precision.
 *
 * Must run right before ConvertPrecision.
 */
class KeepReduceProdI64 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("KeepReduceProdI64");
    KeepReduceProdI64();
};

}  // namespace ov::intel_cpu
