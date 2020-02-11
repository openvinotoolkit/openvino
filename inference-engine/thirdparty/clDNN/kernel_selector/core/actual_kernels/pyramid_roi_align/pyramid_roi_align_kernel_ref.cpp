// Copyright (c) 2018 Intel Corporation
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

#include "pyramid_roi_align_kernel_ref.h"
#include "kernel_selector_utils.h"

#include <vector>

namespace kernel_selector {
ParamsKey PyramidROIAlignKernelRef::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);

    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::byxf);

    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::byxf);

    k.EnableBatching();
    k.EnableDifferentTypes();

    return k;
}

PyramidROIAlignKernelBase::DispatchData PyramidROIAlignKernelRef::SetDefault(const PyramidROIAlign_params& params) const {
    auto dispatch = PyramidROIAlignKernelBase::SetDefault(params);

    std::vector<size_t> global = {
        params.output.X().v * params.output.Y().v,
        params.output.Feature().v,
        params.output.Batch().v };

    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    dispatch.gws0 = global[0];
    dispatch.gws1 = global[1];
    dispatch.gws2 = global[2];

    dispatch.lws0 = local[0];
    dispatch.lws1 = local[1];
    dispatch.lws2 = local[2];

    return dispatch;
}

KernelsData PyramidROIAlignKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options, FORCE_PRIORITY_9);
}
}  // namespace kernel_selector
