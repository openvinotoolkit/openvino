// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// reverse_sequence_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct reverse_sequence_params : public base_params {
    reverse_sequence_params() : base_params(KernelType::REVERSE_SEQUENCE),
    seq_axis(0), batch_axis(0) {}

    int32_t seq_axis;
    int32_t batch_axis;
};

class ReverseSequenceKernelRef : public KernelBaseOpenCL {
public:
    ReverseSequenceKernelRef() : KernelBaseOpenCL("reverse_sequence_ref") {}
    virtual ~ReverseSequenceKernelRef() {}
    virtual JitConstants GetJitConstants(const reverse_sequence_params& params) const;
    virtual CommonDispatchData SetDefault(const reverse_sequence_params& params) const;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
