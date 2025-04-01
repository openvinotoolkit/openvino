// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "col2im_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

bool Col2ImKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::COL2IM) {
        return false;
    }

    const col2im_params& params = static_cast<const col2im_params&>(p);
    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    if (params.inputs[0].Dimentions() > 5)
        return false;

    return true;
}

JitConstants Col2ImKernelBase::GetJitConstants(const col2im_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    auto input = params.inputs[0];
    const auto& output_size = params.output_size;
    const auto& stride = params.stride;
    const auto& dilation = params.dilation;
    const auto& pads_begin = params.padding_begin;
    const auto& pads_end = params.padding_begin;

    const auto orig_height = (output_size.x + pads_begin.x + pads_end.x - (dilation.x * (params.kernel_size.x - 1) + 1)) / stride.x + 1;
    const auto orig_width = (output_size.y + pads_begin.y + pads_end.y - (dilation.y * (params.kernel_size.y - 1) + 1)) / stride.y + 1;
    jit.AddConstant(MakeJitConstant("ORIG_HEIGHT", orig_height));
    jit.AddConstant(MakeJitConstant("ORIG_WIDTH", orig_width));

    // Consider input tensor : (N, C * Product(kernel_size), L)
    const auto num_elements_for_block = input.Feature().v;
    const auto num_blocks = input.Y().v;
    const auto kernel_product = params.kernel_size.x * params.kernel_size.y;
    const auto num_channels = num_elements_for_block / kernel_product;
    jit.AddConstant(MakeJitConstant("NUM_ELEMENTS_FOR_BLOCK", num_elements_for_block));
    jit.AddConstant(MakeJitConstant("KERNEL_PRODUCT", kernel_product));
    jit.AddConstant(MakeJitConstant("NUM_CHANNELS", num_channels));
    jit.AddConstant(MakeJitConstant("NUM_BLOCKS", num_blocks));

    jit.AddConstant(MakeJitConstant("OUT", params.output_size));
    jit.AddConstant(MakeJitConstant("KERNEL", params.kernel_size));
    jit.AddConstant(MakeJitConstant("STRIDE", stride));
    jit.AddConstant(MakeJitConstant("DILATION", dilation));
    jit.AddConstant(MakeJitConstant("PAD_BEGIN", pads_begin));
    jit.AddConstant(MakeJitConstant("PAD_END", pads_end));

    return jit;
}

KernelsData Col2ImKernelBase::GetCommonKernelsData(const Params& params) const {
    KernelData kd = KernelData::Default<col2im_params>(params);
    col2im_params& newParams = *static_cast<col2im_params*>(kd.params.get());

    if (!Validate(params)) {
        return {};
    }

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     EXE_MODE_DEFAULT, false, false, 1, GetFusedPrimitiveInputsCount(params));

    return { kd };
}
}  // namespace kernel_selector
