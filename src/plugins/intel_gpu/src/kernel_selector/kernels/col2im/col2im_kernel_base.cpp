// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "col2im_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

static std::pair<size_t, size_t> get_origin_size(const col2im_params& params) {
    const auto& output_size = params.output_size;
    const auto& stride = params.stride;
    const auto& dilation = params.dilation;
    const auto& pads_begin = params.padding_begin;
    const auto& pads_end = params.padding_begin;

    const auto orig_height = (output_size.x + pads_begin.x + pads_end.x - (dilation.x * (params.kernel_size.x - 1) + 1)) / stride.x + 1;
    const auto orig_width = (output_size.y + pads_begin.y + pads_end.y - (dilation.y * (params.kernel_size.y - 1) + 1)) / stride.y + 1;

    return std::pair(orig_height, orig_width);
}

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

bool Col2ImKernelBase::CheckCol2ImContainBatch(const col2im_params& params) const {
    auto input = params.inputs[0];
    auto orig_size = get_origin_size(params);

    // Check input size L which is the total number of blocks : product from d=1 to 2 of origin size
    if (input.Y().v == 1 && input.Y().v != (orig_size.first * orig_size.second))
        return false;

    return true;
}


JitConstants Col2ImKernelBase::GetJitConstants(const col2im_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    auto input = params.inputs[0];

    auto orig_size = get_origin_size(params);
    jit.AddConstant(MakeJitConstant("ORIG_HEIGHT", orig_size.first));
    jit.AddConstant(MakeJitConstant("ORIG_WIDTH", orig_size.second));

    // Consider input tensor : (N, C * Product(kernel_size), L)
    bool is_batched = CheckCol2ImContainBatch(params);

    const auto num_blocks = is_batched ? input.Y().v : input.Feature().v;

    const size_t num_elements_for_block = is_batched ? input.Feature().v : input.Batch().v;
    const size_t kernel_product = (size_t)(params.kernel_size.x * params.kernel_size.y);
    const size_t num_channels = std::max(num_elements_for_block / kernel_product, (size_t)1);
    jit.AddConstant(MakeJitConstant("NUM_ELEMENTS_FOR_BLOCK", num_elements_for_block));
    jit.AddConstant(MakeJitConstant("KERNEL_PRODUCT", kernel_product));
    jit.AddConstant(MakeJitConstant("NUM_CHANNELS", num_channels));
    jit.AddConstant(MakeJitConstant("NUM_BLOCKS", num_blocks));

    jit.AddConstant(MakeJitConstant("OUT", params.output_size));
    jit.AddConstant(MakeJitConstant("KERNEL", params.kernel_size));
    jit.AddConstant(MakeJitConstant("STRIDE", params.stride));
    jit.AddConstant(MakeJitConstant("DILATION", params.dilation));
    jit.AddConstant(MakeJitConstant("PAD_BEGIN", params.padding_begin));
    jit.AddConstant(MakeJitConstant("PAD_END", params.padding_end));

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
