// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fully_connected_kernel_base.h"
#include "fully_connected_kernel_bf_tiled.h"

#include <vector>

namespace kernel_selector {

class FullyConnected_bf_tiled_dyn_b : public FullyConnectedKernelBase {
public:
    using Parent = FullyConnectedKernelBase;
    FullyConnected_bf_tiled_dyn_b();

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

    struct tune_params {
        tune_params(unsigned tile_ofm, unsigned tile_ifm,
                    unsigned tile_k, unsigned dispatch_bsv, unsigned dispatch_fsv,
                    std::string exec_options)
            : tile_ofm(tile_ofm), tile_ifm(tile_ifm),
              tile_k(tile_k), dispatch_bsv(dispatch_bsv), dispatch_fsv(dispatch_fsv),
              exec_options(exec_options) {}
        tune_params() = default;

        unsigned tile_ofm = 2;
        unsigned tile_ifm = 1;
        unsigned tile_k = 4;
        unsigned dispatch_bsv = 1;
        unsigned dispatch_fsv = 1;
        std::string exec_options;
    };

    // Select optimal TILE_B for the given batch_size.
    // Prefers exact divisor in [8..4]; falls back to 8 with tail handling.
    static size_t select_tile_b(size_t batch_size);

    // Check whether dyn_b is beneficial for the given params (IFM/OFM ratio, !swiglu, INT4 compressed F16).
    // Used by bf_tiled to decide whether to add dyn_b as a sub-kernel.
    static bool IsBeneficial(const fully_connected_params& params);

    // Exposed for bf_tiled integration (sub-kernel creation in GetMultiKernelsData)
    tune_params GetTuneParams(const fully_connected_params& params) const;
    DispatchData SetDefault(const fully_connected_params& params, int autoTuneIndex = -1, int kernel_number = 0) const override;
    JitConstants GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const override;

protected:
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ACTIVATION,
                 FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE };
    }
    bool Validate(const Params& params) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

}  // namespace kernel_selector
