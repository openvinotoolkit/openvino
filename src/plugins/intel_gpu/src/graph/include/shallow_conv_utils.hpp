// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/layout.hpp"

namespace cldnn {

/// A convolution whose input has a low channel count (<=4, e.g. RGB/RGBD) and whose output has
/// enough features to form full f16 feature blocks can directly produce b_fs_yx_fsv16 output via
/// ConvolutionKernel_bfyx_to_bfyx_f16, avoiding an explicit bfyx -> b_fs_yx_fsv16 reorder.
///
/// Shared by the graph passes (select_preferred_formats, layout_optimizer) so the thresholds
/// cannot drift; keep them in sync with the kernel's Validate() (input.Feature() <= 4) and its
/// feature_block_size (16). Callers that also require a specific input format (e.g. bfyx)
/// check that separately.
inline bool is_shallow_conv_fsv16_candidate(const layout& input_layout, const layout& output_layout) {
    return input_layout.feature() <= 4 && output_layout.feature() >= 16;
}

}  // namespace cldnn
