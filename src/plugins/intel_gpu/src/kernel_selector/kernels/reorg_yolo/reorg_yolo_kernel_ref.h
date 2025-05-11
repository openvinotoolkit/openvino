// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// reorg_yolo_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct reorg_yolo_params : public base_params {
    reorg_yolo_params() : base_params(KernelType::REORG_YOLO), stride(0) {}

    uint32_t stride;

    ParamsKey GetParamsKey() const override {
        auto k = base_params::GetParamsKey();
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ReorgYoloKernelRef
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ReorgYoloKernelRef : public KernelBaseOpenCL {
public:
    ReorgYoloKernelRef() : KernelBaseOpenCL("reorg_yolo_gpu_ref") {}
    virtual ~ReorgYoloKernelRef() {}

    using DispatchData = CommonDispatchData;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& params) const override;

protected:
    virtual JitConstants GetJitConstants(const reorg_yolo_params& params) const;
};
}  // namespace kernel_selector
