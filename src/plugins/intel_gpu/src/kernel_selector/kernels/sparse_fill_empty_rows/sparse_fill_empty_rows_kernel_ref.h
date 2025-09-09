// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <kernel_base_opencl.h>

namespace kernel_selector {
struct sparse_fill_empty_rows_params : public base_params {
    sparse_fill_empty_rows_params() : base_params{KernelType::SPARSE_FILL_EMPTY_ROWS} {}
};

class SparseFillEmptyRowsKernelRef : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;

    using DispatchData = CommonDispatchData;

    SparseFillEmptyRowsKernelRef() : KernelBaseOpenCL{"sparse_fill_empty_rows_ref"} {}
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params&) const override;

protected:
    JitConstants GetJitConstants(const sparse_fill_empty_rows_params& params) const;
    bool SkipKernelExecution(const sparse_fill_empty_rows_params& params, size_t kernel_id = 0) const;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

} // namespace kernel_selector
