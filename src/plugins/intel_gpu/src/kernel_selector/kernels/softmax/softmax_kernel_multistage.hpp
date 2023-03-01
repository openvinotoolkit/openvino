// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "softmax_kernel_base.h"

namespace kernel_selector {

class SoftmaxKernel_multistage : public SoftmaxKernelBase {
public:
    SoftmaxKernel_multistage() : SoftmaxKernelBase {"softmax_gpu_multistage"} {}
    ~SoftmaxKernel_multistage() override = default;

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params, const optional_params& options) const override;

protected:
    bool Validate(const Params& params, const optional_params& options) const override;
    JitConstants GetJitConstants(const softmax_params& params, DispatchData dispatchData) const override;

private:
    static constexpr int kElementsPerThread = 4;
    static constexpr int kSubGroupSize = 16;

    static size_t GetDataSetSize(const softmax_params& params);
    static size_t GetDataSetCount(const softmax_params& params);
    SoftmaxKernelBase::DispatchData SetDefault(const softmax_params& params) const;
};

} // namespace kernel_selector
