// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {

/**
 * UniqueCount reference kernel parameters.
 */
struct unique_count_params : base_params {
    unique_count_params() : base_params(KernelType::UNIQUE_COUNT) {}
    bool flattened{};
    int64_t axis{};
};

/**
 * Reference kernel for UniqueCount.
 */
class UniqueCountKernelRef : public KernelBaseOpenCL {
public:
    UniqueCountKernelRef() : KernelBaseOpenCL{"unique_count_ref"} {}
    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const unique_count_params& kernel_params) const;
    static CommonDispatchData SetDefault(const unique_count_params& kernel_params);
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

/**
 * UniqueGather reference kernel parameters.
 */
struct unique_gather_params : base_params {
    unique_gather_params() : base_params(KernelType::UNIQUE_GATHER) {}
    bool flattened{};
    int64_t axis{};
    bool sorted{};
};

/**
 * Reference kernel for UniqueGather.
 */
class UniqueGatherKernelRef : public KernelBaseOpenCL {
public:
    UniqueGatherKernelRef() : KernelBaseOpenCL{"unique_gather_ref"} {}
    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const unique_gather_params& kernel_params) const;
    static CommonDispatchData SetDefault(const unique_gather_params& kernel_params);
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

}  // namespace kernel_selector
