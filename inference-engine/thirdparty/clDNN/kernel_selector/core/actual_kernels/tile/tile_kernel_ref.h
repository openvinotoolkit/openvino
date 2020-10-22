// Copyright (c) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// tile_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct tile_params : public base_params {
    tile_params() : base_params(KernelType::TILE), axis(TileAxis::BATCH), tiles(0) {}

    TileAxis axis;
    int tiles;

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
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
    virtual CommonDispatchData SetDefault(const tile_params& params, const optional_params&) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
