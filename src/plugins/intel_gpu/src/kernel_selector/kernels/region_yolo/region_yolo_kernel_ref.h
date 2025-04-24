// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// region_yolo_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct region_yolo_params : public base_params {
    region_yolo_params() : base_params(KernelType::REGION_YOLO),
    coords(0), classes(0), num(0), mask_size(0), do_softmax(false) {}

    uint32_t coords;
    uint32_t classes;
    uint32_t num;
    uint32_t mask_size;
    bool do_softmax;

    ParamsKey GetParamsKey() const override {
        auto k = base_params::GetParamsKey();
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// RegionYoloKernelRef
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class RegionYoloKernelRef : public KernelBaseOpenCL {
public:
    RegionYoloKernelRef() : KernelBaseOpenCL("region_yolo_gpu_ref") {}
    virtual ~RegionYoloKernelRef() {}

    using DispatchData = CommonDispatchData;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    virtual JitConstants GetJitConstants(const region_yolo_params& params) const;
    bool Validate(const Params& p) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::REORDER };
    }
};
}  // namespace kernel_selector
