// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// bucketize_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct bucketize_params : public base_params {
    bucketize_params() : base_params(KernelType::BUCKETIZE) , output_type(cldnn::data_types::i64), with_right_bound(true) {}

    cldnn::data_types output_type;
    bool with_right_bound;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// bucketize_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct bucketize_optional_params : optional_params {
    bucketize_optional_params() : optional_params(KernelType::BUCKETIZE) {}
};

class BucketizeKernelRef : public KernelBaseOpenCL {
public:
    BucketizeKernelRef() : KernelBaseOpenCL("bucketize_ref") {}
    virtual ~BucketizeKernelRef() = default;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    virtual CommonDispatchData SetDefault(const bucketize_params& params, const optional_params&) const;
    virtual JitConstants GetJitConstants(const bucketize_params& params) const;
    bool Validate(const Params& p, const optional_params& o) const override;
};
}  // namespace kernel_selector
