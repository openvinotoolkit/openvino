// Copyright (c) 2020 Intel Corporation
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


#include "eltwise_kernel_b_fs_yx_fsv16.h"
#include "kernel_selector_utils.h"
#include <string>
#include <algorithm>

namespace kernel_selector {

ParamsKey EltwiseKernel_b_fs_yx_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableTensorPitches();
    k.EnableTensorOffset();
    k.EnableEltwiseBroadcast();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

static inline size_t GetBlockSize(const eltwise_params& params) {
    size_t optimal_bs_values[] = {8, 4, 2, 1};

    for (auto bs : optimal_bs_values) {
        if ((params.output.X().v) % bs == 0) {
            return bs;
        }
    }

    return 1;
}

JitConstants EltwiseKernel_b_fs_yx_fsv16::MakeLoadJitConstants(const eltwise_params& params, bool /*useVload8*/) const {
    JitConstants jit = {};
    std::string vload_decls;
    for (size_t op_num = 0; op_num < params.operations.size(); op_num++) {
        const std::string op_num_str = std::to_string(op_num);
        const auto &ew = params.operations[op_num];
        for (size_t input_idx = 0; input_idx < ew.inputs.size(); input_idx++) {
            const auto &input = ew.inputs[input_idx];
            const std::string name = "INPUT_" + op_num_str + "_" + std::to_string(input_idx);
            std::string idx_order = "INPUT" + std::to_string(input.index) + "_IDX_ORDER";

            switch (input.mode) {
                case EltwiseInputMode::SCALAR:
                    jit.AddConstant(MakeJitConstant(name, input.scalar));
                    break;
                case EltwiseInputMode::INPUT_BUFFER:
                    if (params.inputs[input.index].LogicalSize() == params.output.Feature().v &&
                        params.inputs[input.index].LogicalSize() == params.inputs[input.index].Feature().v) {
                        jit.AddConstant(MakeJitConstant(name,
                                                        "BLOCK_READN(INPUT" + std::to_string(input.index) + "_TYPE, 1, input" + std::to_string(input.index) +
                                                        ", INPUT"+std::to_string(input.index)+"_GET_INDEX(b, f_block*16, y, x))"));
                    } else if (params.inputs[input.index].LogicalSize() == 1) {
                        jit.AddConstant(MakeJitConstant(name,
                                                        "input" + std::to_string(input.index) +
                                                        "[0]"));
                    } else {
                        jit.AddConstant(MakeJitConstant(name,
                                                        "READ_FUNC(input" + std::to_string(input.index) +
                                                        ", INPUT"+std::to_string(input.index)+"_GET_INDEX(b, f_block*16, y, x))"));
                    }
                    break;
                case EltwiseInputMode::OUTPUT_BUFFER:
                    jit.AddConstant(MakeJitConstant(name, "output[off]"));
                    break;
                case EltwiseInputMode::UNORDERED_ACCESS_INPUT_BUFFER:
                    jit.AddConstant(MakeJitConstant(
                            name,
                            "input" + std::to_string(input.index) + "[(size_t)tmp" + std::to_string(input.tmpIndex) + "]"));
                    break;
                case EltwiseInputMode::INTERMEDIATE_RESULTS_INDEX:
                    jit.AddConstant(MakeJitConstant(name, "tmp" + std::to_string(input.tmpIndex)));
                    break;
                default:
                    break;
            }
        }
    }

    return jit;
}

JitConstants EltwiseKernel_b_fs_yx_fsv16::GetJitConstants(const eltwise_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    bool useVload8 = false;

    auto blockSize = GetBlockSize(params);
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jit.AddConstant(MakeJitConstant("BLOCK_SIZE", blockSize));
    jit.AddConstant(MakeJitConstant("BLOCKS_COUNT", CeilDiv(params.output.X().v, blockSize)));

    jit.Merge(MakeInputDeclsJitConstants(params, useVload8));
    jit.Merge(MakeIndexJitConstants(params, useVload8));
    jit.Merge(MakeLoadJitConstants(params, useVload8));
    jit.Merge(GetOperationsJitConstants(params, useVload8, blockSize));

    std::string do_eltwise;
    auto& operations = params.operations;
    for (size_t op_num = 0; op_num < operations.size(); op_num++) {
        do_eltwise += "\\\n\tOPERATION" + std::to_string(op_num) + ";";
    }

    do_eltwise += "\\\n\tres = tmp" + std::to_string(operations.size() - 1) + ";";

    jit.AddConstant(MakeJitConstant("DO_ELTWISE", do_eltwise));

    if (params.layoutBased || params.int8_quantization || params.broadcast) {
        jit.Merge(GetTensorFriendlyWorkGroupsJit(params.output));
    }

    if (!params.stride.empty()) {
        jit.AddConstant(MakeJitConstant("INPUT_STRIDED", 1));
    }

    jit.Merge(MakeActivationJitConstants(params.activations, params.output.GetDType(), "_TYPED"));

    if (params.output.Feature().v % 16 != 0)
        jit.AddConstant(MakeJitConstant("LEFTOVERS", params.output.Feature().v % 16));

    if (!params.fused_ops.empty()) {
        kernel_selector::Datatype input_dt = GetAccumulatorType(params);

        FusedOpsConfiguration conf = {"", {"b", "f_block", "y", "x"}, "res", input_dt, blockSize};
        conf.load_type = FusedOpsConfiguration::LoadType::LT_ALIGNED_READ;
        conf.vec_axis = Tensor::DataChannelName::X;

        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

bool EltwiseKernel_b_fs_yx_fsv16::Validate(const Params& params, const optional_params& o) const {
    if (!EltwiseKernelBase::Validate(params, o)) {
        return false;
    }

    const auto& ewParams = static_cast<const eltwise_params&>(params);

    const auto& output = ewParams.output;
    const auto count = output.PhysicalSize();

    if (count % 8 != 0)
        return false;

    for (size_t i = 0; i < ewParams.inputs.size(); i++) {
        // Allow the same input sizes OR per-channel operation
        if ((ewParams.inputs[i].LogicalSize() != output.LogicalSize()) &&
            (ewParams.inputs[i].LogicalSize() != output.Feature().v || ewParams.inputs[i].Feature().v != output.Feature().v) &&
            (ewParams.inputs[i].LogicalSize() != 1))
            return false;
    }

    auto input0 = ewParams.inputs[0];

    for (size_t i = 1; i < ewParams.inputs.size(); i++) {
        if (input0.GetDType() != ewParams.inputs[i].GetDType()) {
            return false;
        }
    }

    // Check that padding before features doesn't miss-align the blocks
    auto feature_block_size = 16;
    if (input0.Feature().pad.before % feature_block_size != 0 || output.Feature().pad.before % feature_block_size != 0) {
        return false;
    }

    for (size_t i = 1; i < ewParams.inputs.size(); i++) {
        if (ewParams.inputs[i].LogicalSize() == input0.LogicalSize() && !(ewParams.inputs[i] == input0))
            return false;
        if (ewParams.inputs[i].Feature().pad.before % feature_block_size != 0) {
            return false;
        }
    }

    return true;
}

EltwiseKernelBase::DispatchData EltwiseKernel_b_fs_yx_fsv16::SetDefault(const eltwise_params& params) const {
    DispatchData dispatchData;

    dispatchData.gws[0] = Align(params.output.Feature().v, 16);
    dispatchData.gws[1] = CeilDiv(params.output.X().v, GetBlockSize(params)) * params.output.Y().v;
    dispatchData.gws[2] = params.output.Batch().v;

    dispatchData.lws[0] = 16;
    dispatchData.lws[1] = 16;
    while (dispatchData.lws[1] > 1) {
        if (dispatchData.gws[1] % dispatchData.lws[1] == 0)
            break;
        dispatchData.lws[1]--;
    }
    dispatchData.lws[2] = 1;

    dispatchData.efficiency = FORCE_PRIORITY_1;
    return dispatchData;
}

KernelsData EltwiseKernel_b_fs_yx_fsv16::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    KernelData kd = KernelData::Default<eltwise_params>(params);
    eltwise_params& newParams = *static_cast<eltwise_params*>(kd.params.get());

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
    auto cldnn_jit = GetJitConstants(newParams);
    std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

    DispatchData dispatchData = SetDefault(newParams);

    auto& kernel = kd.kernels[0];

    kernel.workGroups.global = dispatchData.gws;
    kernel.workGroups.local = dispatchData.lws;

    kernel.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, DEFAULT);
    kernel.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(),
                                   false,
                                   false,
                                   GetFusedPrimitiveInputsCount(params));

    kd.estimatedTime = dispatchData.efficiency;

    return {kd};
}
}  // namespace kernel_selector
