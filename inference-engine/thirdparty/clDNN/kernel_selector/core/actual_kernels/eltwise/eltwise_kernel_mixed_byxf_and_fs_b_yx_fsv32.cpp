// Copyright (c) 2019 Intel Corporation
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


#include "eltwise_kernel_mixed_byxf_and_fs_b_yx_fsv32.h"
#include "kernel_selector_utils.h"
#include <string>
#include <memory>
#include <vector>

namespace kernel_selector {

namespace {
std::shared_ptr<JitConstant> GetJit_GetIndexForDataLayout(std::string jitName,
                                                          std::string prefix,
                                                          DataLayout dataLayout) {
    std::string jitValue;
    switch (dataLayout) {
        case DataLayout::byxf:
            jitValue += "GET_DATA_INDEX(";
            break;
        case DataLayout::fs_b_yx_fsv32:
            jitValue += "GET_DATA_FS_B_YX_FSV32_INDEX(";
            break;
        default:
            throw std::runtime_error("incorrect data_layout");
    }
    jitValue += prefix + ",b,f,y,x)";

    return MakeJitConstant(jitName, jitValue);
}
}  // namespace
// TODO: [blocked_formats] does fp32 work well with kernel?
ParamsKey EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableEltwiseBroadcast();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

JitConstants EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32::GetJitConstants(const eltwise_params& params) const {
    return GetJitConstantsCommon(params, false);
}

bool EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32::Validate(const Params& params, const optional_params& o) const {
    if (!EltwiseKernelBase::Validate(params, o)) {
        return false;
    }

    const auto& ewParams = static_cast<const eltwise_params&>(params);

    const auto& inputs = ewParams.inputs;
    if (inputs.size() != 2) {
        return false;
    }

    for (auto in : inputs) {
        if (in.GetLayout() != DataLayout::fs_b_yx_fsv32 && in.GetLayout() != DataLayout::byxf)
            return false;
    }

    const auto& input1 = inputs[0];
    const auto& input2 = inputs[1];

    if (input1.Feature().v % 32 != 0 || input2.Feature().v % 32 != 0) {
        return false;
    }

    return true;
}

KernelsData EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32::GetKernelsData(const Params& params,
                                                                       const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<eltwise_params>(params);
    eltwise_params& newParams = *static_cast<eltwise_params*>(kd.params.get());

    std::string jit;

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);

    try {
        auto cldnn_jit = GetJitConstants(newParams);
        cldnn_jit.RemoveConstant("INPUT_0_0");
        cldnn_jit.RemoveConstant("INPUT_0_1");

        cldnn_jit.AddConstants({
            MakeJitConstant("INPUT_0_0", "tmp_input_0"),
            MakeJitConstant("INPUT_0_1", "tmp_input_1"),
            });

        auto input0 = newParams.inputs[0];
        std::vector<size_t> inp0_bfyx = { input0.Batch().v, input0.Feature().v, input0.Y().v, input0.X().v };
        auto input1 = newParams.inputs[1];
        std::vector<size_t> inp1_bfyx = { input1.Batch().v, input1.Feature().v, input1.Y().v, input1.X().v };
        std::vector<std::string> bfyx_str   = { "b", "f0", "y", "x" };
        std::vector<std::string> dims_names = { "BATCH_NUM", "FEATURE_NUM", "SIZE_Y", "SIZE_X" };
        for (size_t dim = 0; dim < inp0_bfyx.size(); dim++) {
            std::string dim_str = bfyx_str[dim];
            std::string jit_str_inp0 = dim_str;
            std::string jit_str_inp1 = dim_str;
            if (inp0_bfyx[dim] > inp1_bfyx[dim]) {
                jit_str_inp1 += " % INPUT1_" + dims_names[dim];
            } else if (inp0_bfyx[dim] < inp1_bfyx[dim]) {
                jit_str_inp0 += " % INPUT0_" + dims_names[dim];
            }
            cldnn_jit.AddConstants({
                MakeJitConstant("INPUT0_DIM_" + dim_str, jit_str_inp0),
                MakeJitConstant("INPUT1_DIM_" + dim_str, jit_str_inp1)
                });
        }

        jit = CreateJit(kernelName, cldnn_jit, entry_point);
    } catch (const std::runtime_error&) {
        return KernelsData();
    }

    auto& kernel = kd.kernels[0];
    size_t x;
    size_t y;
    size_t batches;
    size_t featuresRoundedUp;

    auto dims = newParams.output.LogicalDims();
    if (newParams.output.GetLayout() == DataLayout::fs_b_yx_fsv32) {
        x = dims[0];
        y = dims[1];
        batches = dims[2];
        featuresRoundedUp = (((dims[3] - 1) / 32) + 1) * 32;
    } else {  // byxf
        featuresRoundedUp = (((dims[0] - 1) / 32) + 1) * 32;
        x = dims[1];
        y = dims[2];
        batches = dims[3];
    }

    // in fs_b_yx_fsv32 format we will process 2 features per work item, so reads/writes are done in full writes for
    // fp16
    kernel.workGroups.global = {x, y, (featuresRoundedUp * batches) / 2};

    kernel.workGroups.local = {1, 1, 16};

    kernel.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, DEFAULT);
    kernel.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(), false, false);

    if ((newParams.output.GetLayout() == newParams.inputs[0].GetLayout()) &&
        (newParams.output.GetLayout() ==
         newParams.inputs[1].GetLayout())) {  // There is no need for reordering kernel, better use something more optimal
        kd.estimatedTime = FORCE_PRIORITY_9;
    } else {  // There is need for byxf/fsv32 reordering kernel use this one
        kd.estimatedTime = FORCE_PRIORITY_2;
    }

    return {kd};
}
}  // namespace kernel_selector
