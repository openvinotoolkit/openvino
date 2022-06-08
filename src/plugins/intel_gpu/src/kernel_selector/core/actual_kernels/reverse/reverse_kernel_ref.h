// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "string"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// reverse_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
enum class reverse_mode : uint32_t { index, mask };

struct reverse_params : public base_params {
    reverse_params() : base_params(KernelType::REVERSE) {}

    reverse_mode reverseMode = reverse_mode::index;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// reverse_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct reverse_optional_params : optional_params {
    reverse_optional_params() : optional_params(KernelType::REVERSE) {}
};

class ReverseKernelRef : public KernelBaseOpenCL {
public:
    ReverseKernelRef() : KernelBaseOpenCL("reverse_ref") {}

    virtual ~ReverseKernelRef() {}

    virtual JitConstants GetJitConstants(const reverse_params& params) const;

    virtual CommonDispatchData SetDefault(const reverse_params& params, const optional_params&) const;

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;

    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;

    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
