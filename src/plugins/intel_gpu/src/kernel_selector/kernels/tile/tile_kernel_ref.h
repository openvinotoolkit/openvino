// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// tile_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct tile_params : public base_params {
    tile_params() : base_params(KernelType::TILE) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// tile_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct tile_optional_params : optional_params {
    tile_optional_params() : optional_params(KernelType::TILE) {}
};

class TileKernelRef : public KernelBaseOpenCL {
public:
    TileKernelRef() : KernelBaseOpenCL("tile_ref") {}
    virtual ~TileKernelRef() {}

    virtual JitConstants GetJitConstants(const tile_params& params) const;
    virtual CommonDispatchData SetDefault(const tile_params& params) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    void SetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
