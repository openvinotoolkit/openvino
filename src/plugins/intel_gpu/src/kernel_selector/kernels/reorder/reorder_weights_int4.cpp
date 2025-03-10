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
    } else {
        dispatchData.gws = { CeilDiv(output.LogicalSize(), 2), 1, 1 };
    }
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

bool ReorderWeightsKernelInt4::Validate(const Params& params) const {
    const auto& p = static_cast<const reorder_weights_params&>(params);
    const auto& input = p.input;
    const auto& output = p.output;

    if (input.LogicalSize() != input.OFM().v * input.IFM().v ||
        output.LogicalSize() != output.OFM().v * output.IFM().v) {
        return false;
    }

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
