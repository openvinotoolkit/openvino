/*
// Copyright (c) 2021 Intel Corporation
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

#include "non_max_suppression_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {

ParamsKey NonMaxSuppressionKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

JitConstants NonMaxSuppressionKernelRef::GetJitConstants(const non_max_suppression_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({MakeJitConstant("SORT_RESULT_DESCENDING", params.sort_result_descending),
                      MakeJitConstant("BOX_ENCODING", params.box_encoding)});

    int32_t input_index = 2;
    if (params.has_num_select_per_class) {
        jit.AddConstant(MakeJitConstant("NUM_SELECT_PER_CLASS_IDX", input_index));
        input_index++;
    }

    if (params.has_iou_threshold) {
        jit.AddConstant(MakeJitConstant("IOU_THRESHOLD_IDX", input_index));
    }

    if (params.has_score_threshold) {
        jit.AddConstant(MakeJitConstant("SCORE_THRESHOLD_IDX", input_index));
        input_index++;
    }

    if (params.has_soft_nms_sigma) {
        jit.AddConstant(MakeJitConstant("SOFT_NMS_SIGMA_IDX", input_index));
        input_index++;
    }

    jit.AddConstant(MakeJitConstant("INPUT_ARG_SIZE", input_index));

    return jit;
}

NonMaxSuppressionKernelRef::DispatchData SetDefault(const non_max_suppression_params& params) {
    NonMaxSuppressionKernelRef::DispatchData dispatchData;

    const auto& input = params.inputs[0];
    if (input.GetLayout() == DataLayout::bfyx) {
        dispatchData.gws = {input.Batch().v, 1, 1};
    }
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

bool NonMaxSuppressionKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::NON_MAX_SUPPRESSION || o.GetType() != KernelType::NON_MAX_SUPPRESSION) {
        return false;
    }

    const non_max_suppression_params& params = static_cast<const non_max_suppression_params&>(p);

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

KernelsData NonMaxSuppressionKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    const non_max_suppression_params& orgParams = static_cast<const non_max_suppression_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);
    KernelData kd = KernelData::Default<non_max_suppression_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    printf("JIT: %s\n", jit.c_str());
    printf("orgParams.inputs.size(): %lu\n", orgParams.inputs.size());

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, "", false, false, orgParams.inputs.size());

    return {kd};
}

KernelsPriority NonMaxSuppressionKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
