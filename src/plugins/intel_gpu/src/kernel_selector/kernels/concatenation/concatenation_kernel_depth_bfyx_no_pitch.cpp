// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concatenation_kernel_depth_bfyx_no_pitch.h"
#include "kernel_selector_utils.h"
#include <algorithm>

namespace kernel_selector {

ParamsKey ConcatenationKernel_depth_bfyx_no_pitch::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableTensorOffset();
    k.EnableBatching();
    k.EnableConcatAxis(ConcatAxis::FEATURE);
    k.EnableConcatKernelPerInput();
    return k;
}

DeviceFeaturesKey ConcatenationKernel_depth_bfyx_no_pitch::get_required_device_features_key(const Params& params) const {
    return get_common_subgroups_device_features_key(params);
}

bool ConcatenationKernel_depth_bfyx_no_pitch::Validate(const Params& p) const {
    if (!ConcatenationKernelBase::Validate(p)) {
        return false;
    }

    const concatenation_params& params = static_cast<const concatenation_params&>(p);

    // all inputs have to have same layout
    auto same_layout = params.inputs[0].GetLayout();
    for (const auto& lt : params.inputs) {
        if (lt.GetLayout() != same_layout) {
            return false;
        }
    }

    // kernel uses intel_sub_group_block_read that has 4-byte alignment requirement
    if (params.outputs[0].GetDType() == Datatype::F16) {
        size_t output_offset = 0;

        for (size_t i = 0; i < params.inputs.size(); i++) {
            for (size_t b = 0; b < params.outputs[0].Batch().v; b++) {
                if ((output_offset + b * params.inputs[i].Batch().pitch) % 2 != 0)
                    return false;
            }
            output_offset += params.inputs[i].Batch().pitch;
        }
    }

    return true;
}

ConcatenationKernelBase::DispatchData ConcatenationKernel_depth_bfyx_no_pitch::SetDefault(const concatenation_params& params) const {
    DispatchData dispatchData = ConcatenationKernelBase::SetDefault(params);
    const auto& input = params.inputs[0];
    const auto batch = input.Batch().v;
    dispatchData.gws[0] = batch;
    dispatchData.gws[1] = Align(std::max((size_t)1, input.LogicalSize() / batch), 16 * 8) / 8;
    dispatchData.gws[2] = 1;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 16;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority ConcatenationKernel_depth_bfyx_no_pitch::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}

KernelsData ConcatenationKernel_depth_bfyx_no_pitch::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}
}  // namespace kernel_selector
