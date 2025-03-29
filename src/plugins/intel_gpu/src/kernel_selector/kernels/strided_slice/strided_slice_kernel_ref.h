// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include <vector>

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// strided_slice_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct strided_slice_params : public base_params {
    strided_slice_params() : base_params(KernelType::STRIDED_SLICE) {}

    std::vector<std::vector<int32_t>> striding_params;
    std::vector<uint8_t> begin_mask;
    std::vector<uint8_t> end_mask;
    std::vector<uint8_t> ellipsis_mask;
    std::vector<uint8_t> new_axis_mask;
    std::vector<uint8_t> shrink_axis_mask;
    base_params::ArgType begin_type = base_params::ArgType::Input;
    base_params::ArgType end_type = base_params::ArgType::Input;
    base_params::ArgType stride_type = base_params::ArgType::Input;
    size_t begin_dims = 0;
    size_t end_dims = 0;
    size_t stride_dims = 0;

    uint32_t GetIndexBegin() const {
        uint32_t input_idx = 0;
        if (begin_type == base_params::ArgType::Input) input_idx++;
        return input_idx;
    }

    uint32_t GetIndexEnd() const {
        uint32_t input_idx = GetIndexBegin();
        if (end_type == base_params::ArgType::Input) input_idx++;
        return input_idx;
    }

    uint32_t GetIndexStride() const {
        uint32_t input_idx = GetIndexEnd();
        if (stride_type == base_params::ArgType::Input) input_idx++;
        return input_idx;
    }
};

class StridedSliceKernelRef : public KernelBaseOpenCL {
public:
    StridedSliceKernelRef() : KernelBaseOpenCL("strided_slice_ref") {}
    virtual ~StridedSliceKernelRef() {}
    virtual JitConstants GetJitConstants(const strided_slice_params& params) const;
    virtual CommonDispatchData SetDefault(const strided_slice_params& params) const;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ACTIVATION };
    }
};
}  // namespace kernel_selector
