// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prior_box_kernel_ref.h"

#include <kernel_selector_utils.h>

#include <iostream>

namespace kernel_selector {

namespace {

CommonDispatchData SetDefault(const prior_box_params& params, const optional_params&) {
    kernel_selector::CommonDispatchData dispatchData;
    dispatchData.gws = {params.width, params.height, 1};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    return dispatchData;
}
}  // namespace

KernelsData PriorBoxKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kernel_data = KernelData::Default<prior_box_params>(params);
    const prior_box_params& new_params = dynamic_cast<const prior_box_params&>(*kernel_data.params.get());

    auto dispatch_data = SetDefault(new_params, options);
    auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params, options);
    auto specific_jit = GetJitConstants(new_params);
    auto jit = CreateJit(kernelName, specific_jit, entry_point);
    FillCLKernelData(kernel_data.kernels[0],
                     dispatch_data,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     2);

    KernelsData kernelsData;
    kernelsData.push_back(std::move(kernel_data));
    return kernelsData;
}

KernelsPriority PriorBoxKernelRef::GetKernelsPriority(const Params& /*params*/,
                                                      const optional_params& /*options*/) const {
    return FORCE_PRIORITY_1;
}

ParamsKey PriorBoxKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableBatching();
    return k;
}

bool PriorBoxKernelRef::Validate(const Params& params, const optional_params& optionalParams) const {
    if (params.GetType() != KernelType::PRIOR_BOX || optionalParams.GetType() != KernelType::PRIOR_BOX) {
        return false;
    }

    // output size, image size
    constexpr uint32_t number_of_inputs = 2;
    auto& priorBox8arams = dynamic_cast<const prior_box_params&>(params);
    if (priorBox8arams.inputs.size() != number_of_inputs) {
        return false;
    }

    return true;
}

JitConstants PriorBoxKernelRef::GetJitConstants(const prior_box_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const std::string name = "PRIOR_BOX_";

    jit.AddConstant(MakeJitConstant(name + "MIN_SIZE", params.min_size));
    jit.AddConstant(MakeJitConstant(name + "MAX_SIZE", params.max_size));
    jit.AddConstant(MakeJitConstant(name + "DENSITY", params.density));
    jit.AddConstant(MakeJitConstant(name + "FIXED_RATIO", params.fixed_ratio));
    jit.AddConstant(MakeJitConstant(name + "FIXED_SIZE", params.fixed_size));
    jit.AddConstant(MakeJitConstant(name + "CLIP", params.clip));
    jit.AddConstant(MakeJitConstant(name + "FLIP", params.flip));
    jit.AddConstant(MakeJitConstant(name + "STEP", params.step));
    jit.AddConstant(MakeJitConstant(name + "OFFSET", params.offset));
    jit.AddConstant(MakeJitConstant(name + "SCALE_ALL_SIZES", params.scale_all_sizes));
    jit.AddConstant(MakeJitConstant(name + "MIN_MAX_ASPECT_RATIO_ORDER", params.min_max_aspect_ratios_order));
    jit.AddConstant(MakeJitConstant(name + "ASPECT_RATIO", params.aspect_ratio));
    jit.AddConstant(MakeJitConstant(name + "VARIANCE", params.variance));
    jit.AddConstant(MakeJitConstant(name + "IWI", params.reverse_image_width));
    jit.AddConstant(MakeJitConstant(name + "IHI", params.reverse_image_height));
    jit.AddConstant(MakeJitConstant(name + "STEP_X", params.step_x));
    jit.AddConstant(MakeJitConstant(name + "STEP_Y", params.step_y));
    jit.AddConstant(MakeJitConstant(name + "WIDTH", params.width));
    jit.AddConstant(MakeJitConstant(name + "HEIGHT", params.height));
    jit.AddConstant(MakeJitConstant(name + "NUM_PRIORS_4", params.num_priors_4));
    jit.AddConstant(MakeJitConstant(name + "WIDTHS", params.widths));
    jit.AddConstant(MakeJitConstant(name + "HEIGHTS", params.heights));
    jit.AddConstant(MakeJitConstant(name + "STEP_WIDTHS", params.step_widths));
    jit.AddConstant(MakeJitConstant(name + "STEP_HEIGHTS", params.step_heights));
    jit.AddConstant(MakeJitConstant(name + "IS_CLUSTERED", params.is_clustered));

    return jit;
}

}  // namespace kernel_selector
