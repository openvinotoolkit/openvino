// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "segment_max_kernel_base.h"

namespace kernel_selector {

// Optimized SegmentMax kernel that exploits the sorted (non-decreasing)
// property of segment_ids to binary-search segment boundaries.
// Uses a 2D work distribution: gws[0] = inner_dim_size, gws[1] = num_segments.
class SegmentMaxKernelOpt : public SegmentMaxKernelBase {
public:
    SegmentMaxKernelOpt() : SegmentMaxKernelBase("segment_max_opt") {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    DispatchData SetDefault(const segment_max_params& params) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

}  // namespace kernel_selector
