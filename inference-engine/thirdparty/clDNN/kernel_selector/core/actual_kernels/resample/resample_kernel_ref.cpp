// Copyright (c) 2016-2019 Intel Corporation
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

#include <core/common/kernel_selector_utils.h>
#include "resample_kernel_ref.h"

namespace kernel_selector {

ParamsKey ResampleKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableReampleType(ResampleType::NEAREST_NEIGHBOR);
    k.EnableReampleType(ResampleType::CAFFE_BILINEAR_INTERP);
    k.EnableReampleType(ResampleType::BILINEAR_INTERP);
    return k;
}

KernelsData ResampleKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

JitConstants ResampleKernelRef::GetJitConstants(const resample_params& params) const {
    JitConstants jit = ResampleKernelBase::GetJitConstants(params);

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (DataTensor::ChannelsCount(params.output.GetLayout()) == 4) {
            idx_order = {"batch", "OF_ID", "oy", "ox"};
        } else if (DataTensor::ChannelsCount(params.output.GetLayout()) == 5) {
            idx_order = {"batch", "OF_ID", "oz", "oy", "ox"};
        }

        FusedOpsConfiguration conf = {"", idx_order, "interp_val", params.inputs[0].GetDType(), 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}
}  // namespace kernel_selector
