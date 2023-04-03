// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {

/**
 * Unique reference kernel parameters.
 */
struct unique_params : base_params {
    unique_params() : base_params(KernelType::UNIQUE) {}
    bool flattened{};
    int64_t axis{};
    bool sorted{};
};

/**
 * Unique reference kernel optional parameters.
 */
struct unique_optional_params : optional_params {
    unique_optional_params() : optional_params(KernelType::UNIQUE) {}
};

/**
 * Reference kernel for Unique.
 */
class UniqueKernelRef : public KernelBaseOpenCL {
public:
    UniqueKernelRef() : KernelBaseOpenCL{"unique_ref"} {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& params, const optional_params& options) const override;
    JitConstants GetJitConstants(const unique_params& kernel_params) const;
    static CommonDispatchData SetDefault(const unique_params& kernel_params);
};

}  // namespace kernel_selector
