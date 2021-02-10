// Copyright (c) 2020 Intel Corporation
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

#include "fully_connected_kernel_base.h"

#include <vector>

namespace kernel_selector {

class FullyConnected_bf_tiled : public FullyConnectedKernelBase {
public:
    using Parent = FullyConnectedKernelBase;
    FullyConnected_bf_tiled();

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsData GetTunedKernelsDataByIndex(const Params &params,
                                           const optional_params &options,
                                           const int autoTuneIndex = -1) const override;
    KernelsData GetKernelsDataForAutoTune(const Params& params, const optional_params& options) const override;

    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

    struct tune_params {
        tune_params(unsigned tile_b,
                    unsigned tile_ofm,
                    unsigned tile_ifm,
                    unsigned tile_k,
                    unsigned dispatch_bsv,
                    unsigned dispatch_fsv,
                    std::string exec_options)
            : tile_b(tile_b)
            , tile_ofm(tile_ofm)
            , tile_ifm(tile_ifm)
            , tile_k(tile_k)
            , dispatch_bsv(dispatch_bsv)
            , dispatch_fsv(dispatch_fsv)
            , exec_options(exec_options)
        {}

        tune_params() = default;

        unsigned tile_b;
        unsigned tile_ofm;
        unsigned tile_ifm;
        unsigned tile_k;
        unsigned dispatch_bsv;
        unsigned dispatch_fsv;
        std::string exec_options;
    };

protected:
    DispatchData SetDefault(const fully_connected_params& params, int autoTuneIndex = -1) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ACTIVATION,
                 FusedOpType::ELTWISE,
                 FusedOpType::SCALE,
                 FusedOpType::QUANTIZE };
    }
    JitConstants GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const override;
    bool Validate(const Params& params, const optional_params& options) const override;

    tune_params GetAutoTuneParams(const fully_connected_params& params, int idx = -1) const;

    std::vector<tune_params> auto_tune_params;
};
}  // namespace kernel_selector
