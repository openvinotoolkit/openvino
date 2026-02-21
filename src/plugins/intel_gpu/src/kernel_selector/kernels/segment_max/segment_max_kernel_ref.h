// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "segment_max_kernel_base.h"

namespace kernel_selector {
class SegmentMaxKernelRef : public SegmentMaxKernelBase {
public:
    SegmentMaxKernelRef() : SegmentMaxKernelBase("segment_max_ref") {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
