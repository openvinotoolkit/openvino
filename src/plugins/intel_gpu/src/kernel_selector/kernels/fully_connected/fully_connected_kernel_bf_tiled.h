// Copyright (C) 2018-2025 Intel Corporation
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
                 FusedOpType::QUANTIZE,
                 FusedOpType::SWIGLU };
    }
    JitConstants GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const override;
    bool Validate(const Params& params) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;

    tune_params GetAutoTuneParams(const fully_connected_params& params, KernelType preffered_kernel_type = KernelType::DEFAULT, int idx = -1) const;

    std::vector<tune_params> auto_tune_params;
};

namespace fc_kernel_bf_tiled_utils {
using namespace kernel_selector;
std::pair<size_t, size_t> get_input_bf_size(const fully_connected_params& params);
std::pair<size_t, size_t> get_output_aligned_bf_size(const fully_connected_params& params,
                                                     bool needs_align,
                                                     uint32_t align_b = 1,
                                                     int32_t align_f = 1);
size_t get_scale_group_size(const fully_connected_params& params);
bool is_8bit_asym_wei(const fully_connected_params& params);
bool is_weight_dyn_quantizable(const fully_connected_params& params);
bool is_per_token_dynamic_quantize(const fully_connected_params& params);
size_t get_dynamic_quantize_group_size(const fully_connected_params& params);
bool should_dynamic_quantize(const fully_connected_params& params);
bool is_weight_vertical(const fully_connected_params& params, size_t output_f);
bool is_weight_horizontal(const fully_connected_params& params, size_t output_f);
bool is_weight_small_kn(const fully_connected_params& params, size_t output_f);
bool is_swiglu_fused(const fully_connected_params& params);
bool is_suitable_outer_ofm(const fully_connected_params& params, size_t output_f);
};  // namespace fc_kernel_bf_tiled_utils

}  // namespace kernel_selector
