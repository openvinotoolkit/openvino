// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/**
 * @brief Decomposes OneHot (v1/v16) with non-Constant on_value/off_value.
 *
 * The cldnn one_hot primitive bakes on/off values into the kernel as jit constants,
 * so they must be known at compile time. Rewrite such a OneHot as a mask plus a Select:
 *     OneHot(idx, depth, on, off)  ->  Select(OneHot(idx, depth, true, false), on, off)
 */
class DecomposeOneHotNonConstValues : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DecomposeOneHotNonConstValues");
    DecomposeOneHotNonConstValues();
};

}  // namespace ov::intel_gpu
