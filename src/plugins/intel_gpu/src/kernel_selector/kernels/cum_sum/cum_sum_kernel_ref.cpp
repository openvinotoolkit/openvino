// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cum_sum_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
ParamsKey CumSumKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();
    return k;
}

JitConstants CumSumKernelRef::GetJitConstants(const cum_sum_params& params, DispatchData dispatchData) const {
    auto jits = CumSumKernelBase::GetJitConstants(params, dispatchData);
    auto axis_idx = GetCumSumAxisIndex(params);

    jits.AddConstant(MakeJitConstant("AXIS_LAYOUT_INDEX", axis_idx));

    if (params.reverse) {
        const auto& output = params.outputs[0];
        if (output.is_dynamic()) {
            const int rank = static_cast<int>(output.LogicalDims().size());
            auto idx = rank - axis_idx - 1;
            int shape_info_idx = idx;
            if (idx >= 2) {
                shape_info_idx += (static_cast<int>(DataTensor::max_rank()) - rank);
            }

            size_t num_of_dynamic_inputs = 0;
            for (auto& input : params.inputs) {
                if (input.is_dynamic()) {
                    num_of_dynamic_inputs += 1;
                }
            }

            jits.AddConstant(MakeJitConstant("STOP_IND", toShapeInfoString(0, shape_info_idx, true, num_of_dynamic_inputs)));
        }
    }

    return jits;
}

KernelsData CumSumKernelRef::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority CumSumKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
