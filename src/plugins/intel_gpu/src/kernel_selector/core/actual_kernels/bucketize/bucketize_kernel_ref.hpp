// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "kernel_base_opencl.h"

namespace kernel_selector {

/**
 * Bucketize reference kernel parameters.
 */
struct bucketize_params : base_params {
    bucketize_params() : base_params(KernelType::BUCKETIZE) {}
    bool with_right_bound = true;
};

/**
 * Bucketize reference kernel optional parameters.
 */
struct bucketize_optional_params : optional_params {
    bucketize_optional_params() : optional_params(KernelType::BUCKETIZE) {}
};

/**
 * Reference kernel for Bucketize.
 */
class BucketizeKernelRef : public KernelBaseOpenCL {
public:
    BucketizeKernelRef() : KernelBaseOpenCL{"bucketize_ref"} {}

private:
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& params, const optional_params& options) const override;
    JitConstants GetJitConstants(const bucketize_params& kernel_params) const;
};

}  // namespace kernel_selector
