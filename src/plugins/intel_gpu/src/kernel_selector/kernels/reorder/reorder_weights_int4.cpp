// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_weights_int4.h"
#include "kernel_selector_common.h"
#include "kernel_selector_params.h"
#include "kernel_selector_utils.h"
#include "common_types.h"

namespace kernel_selector {

ParamsKey ReorderWeightsKernelInt4::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputWeightsType(WeightsType::INT4);
    k.EnableInputWeightsType(WeightsType::UINT4);
    k.EnableOutputWeightsType(WeightsType::UINT4);
    k.EnableOutputWeightsType(WeightsType::INT4);
    k.EnableInputWeightsLayout(WeightsLayout::oiyx);
    k.EnableInputWeightsLayout(WeightsLayout::ioyx);
    k.EnableOutputWeightsLayout(WeightsLayout::os_iyx_osv16);
    k.EnableOutputWeightsLayout(WeightsLayout::os_iyx_osv32);
    k.EnableOutputWeightsLayout(WeightsLayout::os_is_yx_osv32_isv2);
    k.EnableOutputWeightsLayout(WeightsLayout::os_iyx_osv64);
    k.EnableOutputWeightsLayout(WeightsLayout::os_is_yx_osv64_isv2);
    k.EnableOutputWeightsLayout(WeightsLayout::oiyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    return k;
}

KernelsData ReorderWeightsKernelInt4::GetKernelsData(const Params& params) const {
    const reorder_weights_params& orgParams = static_cast<const reorder_weights_params&>(params);
    return GetCommonKernelsData(orgParams);
}

ReorderWeightsKernelInt4::DispatchData ReorderWeightsKernelInt4::SetDefault(const reorder_weights_params& params) const {
    DispatchData dispatchData;

    const auto& output = params.output;

    // Divide one of the dimensions by 2 to save with byte granularity
    if (output.GetLayout() == WeightsLayout::os_iyx_osv32) {
        dispatchData.gws = { Align(output.OFM().v, 32) / 2, output.IFM().v, 1 };
    } else if (output.GetLayout() == WeightsLayout::os_is_yx_osv32_isv2) {
        dispatchData.gws = { Align(output.OFM().v, 32), output.IFM().v / 2, 1 };
    } else if (output.GetLayout() == WeightsLayout::os_iyx_osv16) {
        dispatchData.gws = { Align(output.OFM().v, 16), output.IFM().v / 2, 1 };
    } else if (output.GetLayout() == WeightsLayout::os_iyx_osv64) {
        dispatchData.gws = { Align(output.OFM().v, 64) / 2, output.IFM().v, 1 };
    } else if (output.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2) {
        dispatchData.gws = { Align(output.OFM().v, 64), output.IFM().v / 2, 1 };
    } else if (output.GetLayout() == WeightsLayout::oiyx) {
        auto dims = output.GetDims();
        bool has_pads = std::any_of(dims.begin(), dims.end(), [](const kernel_selector::Tensor::Dim& d) {
            return d.pad.Total() != 0;
        });
        if (has_pads) {
            dispatchData.gws = { CeilDiv(output.PhysicalSize(), 2), 1, 1 };
        } else {
            dispatchData.gws = { CeilDiv(output.LogicalSize(), 2), 1, 1 };
        }
    }
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

JitConstants ReorderWeightsKernelInt4::GetJitConstants(const reorder_weights_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);
    const auto& input = params.input;
    const auto& output = params.output;

    if (input.GetLayout() == WeightsLayout::oiyx && output.GetLayout() == WeightsLayout::oiyx) {
        const auto idims = input.GetDims();
        const auto odims = output.GetDims();

        size_t input_inner_most_idx  = input.GetDims().size() - params.original_input_rank;
        size_t output_inner_most_idx = output.GetDims().size() - params.original_output_rank;

        const auto input_inner_most_dim = idims.at(input_inner_most_idx).LogicalDimPadded();
        const auto output_inner_most_dim = odims.at(output_inner_most_idx).LogicalDimPadded();
        OPENVINO_ASSERT(input_inner_most_dim % 2 != 0 && output_inner_most_dim % 2 == 0,
                        "Reorder weight i4 kernel for data padding only supports"
                        "an odd input innermost dimension and an even output innermost dimension.");
        jit.AddConstant(MakeJitConstant("INPUT0_INNERMOST_NUM", input_inner_most_dim));
        jit.AddConstant(MakeJitConstant("OUTPUT_INNERMOST_NUM", output_inner_most_dim));
    }

    return jit;
}

bool ReorderWeightsKernelInt4::Validate(const Params& params) const {
    const auto& p = static_cast<const reorder_weights_params&>(params);
    const auto& input = p.input;
    const auto& output = p.output;

    // To use the reorder weight i4 kernel for adding padding to an odd innermost dimension,
    // the input tensor should have an odd innermost dimension without any padding,
    // and the output tensor should have padding with only the pad.after value for the innermost dimension.
    if (input.GetLayout() == WeightsLayout::oiyx && output.GetLayout() == WeightsLayout::oiyx) {
        const auto idims = input.GetDims();
        const auto odims = output.GetDims();

        size_t input_inner_most_idx  = idims.size() - p.original_input_rank;
        size_t output_inner_most_idx = odims.size() - p.original_output_rank;

        bool has_pads_for_input_dims = std::any_of(idims.begin(), idims.end(), [](const kernel_selector::Tensor::Dim& d) {
            return d.pad.Total() != 0;
        });
        bool has_pads_for_output_dims_except_inner_most = std::any_of(odims.begin() + output_inner_most_idx + 1, odims.end(),
            [](const kernel_selector::Tensor::Dim& d) {
            return d.pad.Total() != 0;
        });

        if (idims[input_inner_most_idx].v % 2 != 0
            && !has_pads_for_input_dims
            && !has_pads_for_output_dims_except_inner_most
            && odims[output_inner_most_idx].pad.before == 0
            && odims[output_inner_most_idx].LogicalDimPadded() % 2 == 0) {
            return true;
        }
    }

    OPENVINO_ASSERT((input.LogicalSize() == input.OFM().v * input.IFM().v
                    && output.LogicalSize() == output.OFM().v * output.IFM().v),
                    "Reorder weight i4 only supports 2D input/output, except when adding padding for the same shape(WeightsLayout::oiyx).");

    bool supported_case = input.GetLayout() == WeightsLayout::oiyx && output.GetLayout() == WeightsLayout::os_iyx_osv32;
    supported_case |= input.GetLayout() == WeightsLayout::oiyx && output.GetLayout() == WeightsLayout::os_is_yx_osv32_isv2;
    supported_case |= input.GetLayout() == WeightsLayout::oiyx && output.GetLayout() == WeightsLayout::os_iyx_osv16;
    supported_case |= input.GetLayout() == WeightsLayout::oiyx && output.GetLayout() == WeightsLayout::os_iyx_osv64;
    supported_case |= input.GetLayout() == WeightsLayout::oiyx && output.GetLayout() == WeightsLayout::os_is_yx_osv64_isv2;
    supported_case |= input.GetLayout() == WeightsLayout::ioyx && output.GetLayout() == WeightsLayout::oiyx;
    supported_case |= input.GetLayout() == WeightsLayout::ioyx && output.GetLayout() == WeightsLayout::os_iyx_osv32;
    return supported_case;
}

KernelsPriority ReorderWeightsKernelInt4::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
