// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fully_connected_kernel_base.h"

#include <vector>

namespace kernel_selector {

class FullyConnected_bf_tiled : public FullyConnectedKernelBase {
public:
    enum class KernelType : uint8_t {
        DEFAULT = 0,
        SLM,
        ANY
    };

    using Parent = FullyConnectedKernelBase;
    FullyConnected_bf_tiled();

    KernelsData GetKernelsData(const Params& params) const override;
    using FullyConnectedKernelBase::GetTunedKernelsDataByIndex;
    KernelsData GetTunedKernelsDataByIndex(const Params &params,
                                           const int autoTuneIndex = -1) const override;
    KernelsData GetKernelsDataForAutoTune(const Params& params) const override;

    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

    KernelsData GetMultiKernelsData(const Params &params,
                                     DataLayout dl,
                                     WeightsLayout wl,
                                     const std::string exeMode = EXE_MODE_DEFAULT,
                                     int autoTuneIndex = -1) const;

    struct tune_params {
        tune_params(unsigned tile_b,
                    unsigned tile_ofm,
                    unsigned tile_ifm,
                    unsigned tile_k,
                    unsigned outer_ofm,
                    unsigned dispatch_bsv,
                    unsigned dispatch_fsv,
                    std::string exec_options,
                    KernelType kernel_type = KernelType::DEFAULT)
            : tile_b(tile_b)
            , tile_ofm(tile_ofm)
            , tile_ifm(tile_ifm)
            , tile_k(tile_k)
            , outer_ofm(outer_ofm)
            , dispatch_bsv(dispatch_bsv)
            , dispatch_fsv(dispatch_fsv)
            , exec_options(exec_options)
            , kernel_type(kernel_type)
        {}

        tune_params() = default;

        unsigned tile_b;
        unsigned tile_ofm;
        unsigned tile_ifm;
        unsigned tile_k;
        unsigned outer_ofm;
        unsigned dispatch_bsv;
        unsigned dispatch_fsv;
        std::string exec_options;
        KernelType kernel_type;
    };

protected:
    DispatchData SetDefault(const fully_connected_params& params, int autoTuneIndex = -1, int kernel_number = 0) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ACTIVATION,
                 FusedOpType::ELTWISE,
                 FusedOpType::QUANTIZE };
    }
    JitConstants GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const override;
    bool Validate(const Params& params) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;

    tune_params GetAutoTuneParams(const fully_connected_params& params, KernelType preffered_kernel_type = KernelType::DEFAULT, int idx = -1) const;

    std::vector<tune_params> auto_tune_params;
};
}  // namespace kernel_selector
