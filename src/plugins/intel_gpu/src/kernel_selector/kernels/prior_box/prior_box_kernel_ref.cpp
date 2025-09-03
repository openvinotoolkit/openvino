// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prior_box_kernel_ref.h"

#include <kernel_selector_utils.h>

#include <iostream>

namespace kernel_selector {

namespace {

CommonDispatchData SetDefault(const prior_box_params& params) {
    kernel_selector::CommonDispatchData dispatchData;
    dispatchData.gws = {params.width, params.height, 1};
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    return dispatchData;
}
}  // namespace

KernelsData PriorBoxKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kernel_data = KernelData::Default<prior_box_params>(params);
    const prior_box_params& new_params = dynamic_cast<const prior_box_params&>(*kernel_data.params.get());

    const auto dispatch_data = SetDefault(new_params);
    const auto entry_point = GetEntryPoint(kernelName, new_params.layerID, params);
    const auto specific_jit = GetJitConstants(new_params);
    const auto jit = CreateJit(kernelName, specific_jit, entry_point);
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

KernelsPriority PriorBoxKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

ParamsKey PriorBoxKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableDifferentTypes();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableBatching();
    k.EnableTensorPitches();
    return k;
}

bool PriorBoxKernelRef::Validate(const Params& params) const {
    if (params.GetType() != KernelType::PRIOR_BOX) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    const auto& priorBoxParams = dynamic_cast<const prior_box_params&>(params);
    if (priorBoxParams.inputs.size() != 2) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    // Current ref kernel doesn't support clustered version
    if (priorBoxParams.is_clustered) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }
    return true;
}

JitConstants PriorBoxKernelRef::GetJitConstants(const prior_box_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("MIN_SIZE", params.min_size));
    jit.AddConstant(MakeJitConstant("MAX_SIZE", params.max_size));
    jit.AddConstant(MakeJitConstant("DENSITY", params.density));
    jit.AddConstant(MakeJitConstant("FIXED_RATIO", params.fixed_ratio));
    jit.AddConstant(MakeJitConstant("FIXED_SIZE", params.fixed_size));
    if (params.clip) {
        jit.AddConstant(MakeJitConstant("CLIP", 1));
    }
    if (params.flip) {
        jit.AddConstant(MakeJitConstant("FLIP", 1));
    }
    if (params.step != 0.0f) {
        jit.AddConstant(MakeJitConstant("STEP", params.step));
    }
    jit.AddConstant(MakeJitConstant("OFFSET", params.offset));
    jit.AddConstant(MakeJitConstant("SCALE_ALL_SIZES", params.scale_all_sizes));
    if (params.min_max_aspect_ratios_order) {
        jit.AddConstant(MakeJitConstant("MIN_MAX_ASPECT_RATIO_ORDER", 1));
    }
    jit.AddConstant(MakeJitConstant("ASPECT_RATIO", params.aspect_ratio));
    jit.AddConstant(MakeJitConstant("VARIANCE", params.variance));
    jit.AddConstant(MakeJitConstant("IWI", params.reverse_image_width));
    jit.AddConstant(MakeJitConstant("IHI", params.reverse_image_height));
    jit.AddConstant(MakeJitConstant("STEP_X", params.step_x));
    jit.AddConstant(MakeJitConstant("STEP_Y", params.step_y));
    jit.AddConstant(MakeJitConstant("WIDTH", params.width));
    jit.AddConstant(MakeJitConstant("HEIGHT", params.height));
    jit.AddConstant(MakeJitConstant("NUM_PRIORS_4", params.num_priors_4));
    jit.AddConstant(MakeJitConstant("WIDTHS", params.widths));
    jit.AddConstant(MakeJitConstant("HEIGHTS", params.heights));

    return jit;
}

}  // namespace kernel_selector
