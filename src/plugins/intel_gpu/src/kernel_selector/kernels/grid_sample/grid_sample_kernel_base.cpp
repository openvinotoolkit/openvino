// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "grid_sample_kernel_base.hpp"

#include "kernel_selector_utils.h"

namespace kernel_selector {

KernelsData GridSampleKernelBase::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    auto kernel_data = KernelData::Default<grid_sample_params>(params);
    const auto& kernel_params = dynamic_cast<const grid_sample_params&>(*kernel_data.params);
    const auto dispatch_data = CalcDispatch(kernel_params);
    const auto entry_point = GetEntryPoint(kernelName, kernel_params.layerID, params);
    const auto jit_constants = GetJitConstants(kernel_params);
    const auto jit = CreateJit(kernelName, jit_constants, entry_point);
    auto& kernel = kernel_data.kernels.front();

    FillCLKernelData(kernel, dispatch_data, params.engineInfo, kernelName, jit, entry_point, {}, false, false, 2);

    return {kernel_data};
}

bool GridSampleKernelBase::Validate(const Params& params) const {
    if (params.GetType() != KernelType::GRID_SAMPLE) {
        return false;
    }

    const auto& kernel_params = dynamic_cast<const grid_sample_params&>(params);
    if (kernel_params.inputs.size() != 2) {
        return false;
    }

    return true;
}

JitConstants GridSampleKernelBase::GetJitConstants(const grid_sample_params& kernel_params) const {
    auto jit_constants = MakeBaseParamsJitConstants(kernel_params);

    jit_constants.AddConstants({
        MakeJitConstant("INTERPOLATION_MODE_" + ov::as_string(kernel_params.interpolation_mode), true),
        MakeJitConstant("PADDING_MODE_" + ov::as_string(kernel_params.padding_mode), true),
    });

    if (kernel_params.align_corners) {
        jit_constants.AddConstant(MakeJitConstant("ALIGN_CORNERS", true));
    }

    return jit_constants;
}

}  // namespace kernel_selector

namespace ov {

template <>
ov::EnumNames<kernel_selector::grid_sample_params::InterpolationMode>& ::ov::EnumNames<
    kernel_selector::grid_sample_params::InterpolationMode>::get() {
    static auto enum_names = EnumNames<kernel_selector::grid_sample_params::InterpolationMode>(
        "kernel_selector::grid_sample_params::InterpolationMode",
        {
            {"BILINEAR", kernel_selector::grid_sample_params::InterpolationMode::BILINEAR},
            {"BICUBIC", kernel_selector::grid_sample_params::InterpolationMode::BICUBIC},
            {"NEAREST", kernel_selector::grid_sample_params::InterpolationMode::NEAREST},
        });
    return enum_names;
}

template <>
ov::EnumNames<kernel_selector::grid_sample_params::PaddingMode>& ::ov::EnumNames<
    kernel_selector::grid_sample_params::PaddingMode>::get() {
    static auto enum_names = EnumNames<kernel_selector::grid_sample_params::PaddingMode>(
        "kernel_selector::grid_sample_params::PaddingMode",
        {
            {"ZEROS", kernel_selector::grid_sample_params::PaddingMode::ZEROS},
            {"BORDER", kernel_selector::grid_sample_params::PaddingMode::BORDER},
            {"REFLECTION", kernel_selector::grid_sample_params::PaddingMode::REFLECTION},
        });
    return enum_names;
}

}  // namespace ov
