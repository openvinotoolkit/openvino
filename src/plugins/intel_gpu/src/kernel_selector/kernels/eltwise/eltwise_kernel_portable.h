// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eltwise_kernel_ref.h"

namespace kernel_selector {

// Portable dense traversal with the existing reference operation/type contract.
class EltwiseKernelPortable : public EltwiseKernelRef {
public:
    EltwiseKernelPortable() : EltwiseKernelRef("eltwise_portable") {}

    JitConstants GetJitConstants(const eltwise_params& params) const override;

protected:
    DispatchData SetDefault(const eltwise_params& params) const override;

private:
    bool IsDense(const eltwise_params& params) const;
};

}  // namespace kernel_selector
