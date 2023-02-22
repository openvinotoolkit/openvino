// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
struct adaptive_pooling_params : public base_params {
    adaptive_pooling_params() : base_params(KernelType::ADAPTIVE_POOLING) {}

    PoolType mode{PoolType::MAX};
    Datatype poolIndexElementType = Datatype::INT64;
};

struct adaptive_pooling_optional_params : public optional_params {
    adaptive_pooling_optional_params() : optional_params(KernelType::ADAPTIVE_POOLING) {}
};

class AdaptivePoolingRef : public KernelBaseOpenCL {
public:
    AdaptivePoolingRef() : KernelBaseOpenCL("adaptive_pooling_gpu_ref") {}
    ~AdaptivePoolingRef() = default;

    using DispatchData = CommonDispatchData;

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
protected:
    bool Validate(const Params& p, const optional_params& o) const override;
};
}  // namespace kernel_selector
