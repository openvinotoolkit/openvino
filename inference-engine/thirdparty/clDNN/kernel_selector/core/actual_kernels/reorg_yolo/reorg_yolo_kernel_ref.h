/*
// Copyright (c) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

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

    virtual ParamsKey GetParamsKey() const {
        auto k = base_params::GetParamsKey();
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// reorg_yolo_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct reorg_yolo_optional_params : optional_params {
    reorg_yolo_optional_params() : optional_params(KernelType::REORG_YOLO) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ReorgYoloKernelRef
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ReorgYoloKernelRef : public KernelBaseOpenCL {
public:
    ReorgYoloKernelRef() : KernelBaseOpenCL("reorg_yolo_gpu_ref") {}
    virtual ~ReorgYoloKernelRef() {}

    using DispatchData = CommonDispatchData;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    virtual JitConstants GetJitConstants(const reorg_yolo_params& params) const;
};
}  // namespace kernel_selector
