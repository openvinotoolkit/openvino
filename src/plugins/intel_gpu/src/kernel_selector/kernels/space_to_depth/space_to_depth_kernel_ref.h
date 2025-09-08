// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// space_to_depth_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct space_to_depth_params : public base_params {
    space_to_depth_params() : base_params(KernelType::SPACE_TO_DEPTH), depth_mode(SpaceToDepthMode::BLOCKS_FIRST), block_size(1) {}

    SpaceToDepthMode depth_mode;

    size_t block_size;
};

class SpaceToDepthKernelRef : public KernelBaseOpenCL {
public:
    SpaceToDepthKernelRef() : KernelBaseOpenCL("space_to_depth_ref") {}
    virtual ~SpaceToDepthKernelRef() = default;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    virtual CommonDispatchData SetDefault(const space_to_depth_params& params) const;
    virtual JitConstants GetJitConstants(const space_to_depth_params& params) const;
    bool Validate(const Params& p) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE,
                 FusedOpType::ACTIVATION };
    }
};
}  // namespace kernel_selector
