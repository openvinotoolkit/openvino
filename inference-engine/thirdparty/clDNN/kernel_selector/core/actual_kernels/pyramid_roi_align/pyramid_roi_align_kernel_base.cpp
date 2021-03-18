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

#include "pyramid_roi_align_kernel_base.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {

JitConstants PyramidROIAlignKernelBase::GetJitConstants(const PyramidROIAlign_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("IMAGE_SIZE_X", params.image_size_x));
    jit.AddConstant(MakeJitConstant("IMAGE_SIZE_Y", params.image_size_y));
    jit.AddConstant(MakeJitConstant("SAMPLING_RATIO_X", params.sampling_ratio_x));
    jit.AddConstant(MakeJitConstant("SAMPLING_RATIO_Y", params.sampling_ratio_y));
    jit.AddConstant(MakeJitConstant("PYRAMID_STARTING_LEVEL", params.pyramid_starting_level));

    return jit;
}

PyramidROIAlignKernelBase::DispatchData PyramidROIAlignKernelBase::SetDefault(const PyramidROIAlign_params& params) const {
    DispatchData dispatchData;
    dispatchData.gws = {1, 1, 1};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    return dispatchData;
}

KernelsData PyramidROIAlignKernelBase::GetCommonKernelsData(const Params& params,
                                                            const optional_params& options) const {
    assert(params.GetType() == KernelType::PYRAMID_ROI_ALIGN);

    const auto& prim_params =
        static_cast<const PyramidROIAlign_params&>(params);
    auto dispatchData = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<PyramidROIAlign_params>(params);
    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     (uint32_t)prim_params.inputs.size());

    return {k_data};
}
}  // namespace kernel_selector
