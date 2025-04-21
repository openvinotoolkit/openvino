// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lora_kernel_base.h"

namespace kernel_selector {

class LoRAKernelOpt : public LoRAKernelBase {
public:
    using Parent = LoRAKernelBase;

    struct LoRATuningData {
        size_t regM = 0;
        size_t regN = 0;
        size_t sgM = 0;
        size_t sgN = 0;
        size_t max_gemma_sgk = 0;
        size_t subgroup_size = 0;
    };

    LoRAKernelOpt() : LoRAKernelBase("lora_opt") {}
    virtual ~LoRAKernelOpt() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& params) const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
    JitConstants GetJitConstants(const lora_params& params, size_t kernel_idx) const;
    CommonDispatchData SetDefault(const lora_params& params, size_t kernel_idx) const;
    LoRATuningData GetTuningParams(const lora_params& params, size_t kernel_idx) const;
    std::pair<size_t, size_t> GetSuitableKernels(const lora_params& params) const;

    std::string GenerateBlockRead(Datatype dtype, std::string input) const;
    std::string GenerateBlockWrite(Datatype dtype, std::string dst, std::string src) const;
    std::string GenerateBroadcast(Datatype dtype, std::string input) const;
    std::string GenerateMatMulCode(size_t M, size_t N, Datatype dtype, bool is_A_kernel) const;
    std::string GenerateStoreResultCode(size_t M, size_t N, Datatype dtype, bool is_A_kernel) const;
};

}  // namespace kernel_selector
