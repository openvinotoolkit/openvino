/*
// Copyright (c) 2020 Intel Corporation
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

#include "extract_image_patches_kernel_base.h"
#include <vector>
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey ExtractImagePatchesKernelBase::GetSupportedKey() const {
    ParamsKey k;

    k.EnableAllInputDataType();
    k.EnableAllOutputDataType();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants ExtractImagePatchesKernelBase::GetJitConstants(const extract_image_patches_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("SIZE_ROWS", params.sizes[0]),
        MakeJitConstant("SIZE_COLS", params.sizes[1]),
        MakeJitConstant("STRIDE_ROWS", params.strides[0]),
        MakeJitConstant("STRIDE_COLS", params.strides[1]),
        MakeJitConstant("RATES_ROWS", params.rates[0]),
        MakeJitConstant("RATES_COLS", params.rates[1]),
    });
    if (params.auto_pad == "same_upper")
        jit.AddConstant(MakeJitConstant("AUTO_PAD", 1));
    else if (params.auto_pad == "same_lower")
        jit.AddConstant(MakeJitConstant("AUTO_PAD", 2));

    return jit;
}

ExtractImagePatchesKernelBase::DispatchData ExtractImagePatchesKernelBase::SetDefault(const extract_image_patches_params& params) const {
    DispatchData kd;

    std::vector<size_t> global = { params.output.Batch().v,
                                   params.output.Feature().v,
                                   params.output.Y().v * params.output.X().v };

    const auto& local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    return kd;
}

KernelsData ExtractImagePatchesKernelBase::GetCommonKernelsData(const Params& params,
                                                                const optional_params& options,
                                                                float estimated_time) const {
    if (!Validate(params, options)) {
        return KernelsData();
    }

    const auto& prim_params = static_cast<const extract_image_patches_params&>(params);

    auto run_info = SetDefault(prim_params);
    KernelData kd = KernelData::Default<extract_image_patches_params>(params);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, run_info, params.engineInfo, kernelName, jit, entry_point);

    kd.estimatedTime = estimated_time;

    return {kd};
}

bool ExtractImagePatchesKernelBase::Validate(const Params& p, const optional_params&) const {
    const extract_image_patches_params& params = static_cast<const extract_image_patches_params&>(p);

    if (params.GetType() != KernelType::EXTRACT_IMAGE_PATCHES) {
        return false;
    }

    return true;
}
}  // namespace kernel_selector
