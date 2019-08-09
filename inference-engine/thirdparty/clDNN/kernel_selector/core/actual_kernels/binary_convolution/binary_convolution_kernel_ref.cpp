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


#include "binary_convolution_kernel_ref.h"
#include <string>

namespace kernel_selector {

ParamsKey BinaryConvolutionKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::BINARY);
    k.EnableInputWeightsType(WeightsType::BINARY);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::BINARY);
    k.EnableInputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSplitSupport();
    return k;
}

BinaryConvolutionKernelBase::DispatchData BinaryConvolutionKernelRef::SetDefault(
    const binary_convolution_params& params,
    int) const {
    DispatchData kd = BinaryConvolutionKernelBase::SetDefault(params);

    const auto& out = params.output;

    auto b = out.Batch().v;
    auto f = out.Feature().v;
    auto y = out.Y().v;
    auto x = out.X().v;

    kd.gws0 = b;
    kd.gws1 = f;
    kd.gws2 = x * y;

    kd.lws0 = 1;
    kd.lws1 = 1;
    kd.lws2 = 1;

    kd.effiency = DONT_USE_IF_HAVE_SOMETHING_ELSE;

    return kd;
}

JitConstants BinaryConvolutionKernelRef::GetJitConstants(const binary_convolution_params& params,
                                                         const DispatchData& runInfo) const {
    auto jit = Parent::GetJitConstants(params, runInfo);

    int pad_physical_val = params.pad_value == -1.0f ? 0x00000000 : 0xFFFFFFFF;
    int leftovers_mask = (0xFFFFFFFF >> (32 - params.inputs[0].Feature().v % 32));
    jit.AddConstant(MakeJitConstant("INPUT0_FEATURE_NUM_PACKED", CeilDiv(params.inputs[0].Feature().v, 32)));
    jit.AddConstant(MakeJitConstant("FEATURE_PACK_SIZE", 32));
    jit.AddConstant(MakeJitConstant("OFM_BLOCK_SIZE", 32));
    jit.AddConstant(MakeJitConstant("EXCLUDE_PAD", params.pad_value == 0.0f));
    jit.AddConstant(MakeJitConstant("PAD_VALUE", pad_physical_val));
    jit.AddConstant(MakeJitConstant("LEFTOVERS", params.inputs[0].Feature().v % 32 != 0));
    jit.AddConstant(MakeJitConstant("LEFTOVERS_MASK", leftovers_mask));

    return jit;
}

KernelsData BinaryConvolutionKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}

bool BinaryConvolutionKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (!BinaryConvolutionKernelBase::Validate(p, o) || !CovolutionBinaryCheckInput(p, o))
        return false;

    const auto& params = static_cast<const binary_convolution_params&>(p);

    if (!params.fused_ops.empty())
        return false;

    return true;
}

JitConstants BinaryConvolutionKernelRef::GetFusedPrimitivesJitConstants(const binary_convolution_params& params,
                                                                        const DispatchData& /*kd*/) const {
    JitConstants jit = {};

    size_t op_id = 0;
    std::string input_decls = "";
    std::string eltwise_fused_ops = "";
    for (auto& fused_dep : params.fused_ops) {
        std::string op_type = "";
        switch (fused_dep.type) {
            case binary_convolution_params::fused_operation_desc::Type::SCALE: {
                op_type = "scale";
                // Variables that are supposed to be defined:
                // f (int) - index of output feature channel
                // res (float, half) - results of layer without any fusions
                if (fused_dep.tensors.size() == 1) {
                    eltwise_fused_ops += "res = (res*" + op_type + "_input0[f]);";
                } else {
                    eltwise_fused_ops += "res = (res*" + op_type + "_input0[f] + " + op_type + "_input1[f]);";
                }
                break;
            }
            default:
                throw std::invalid_argument("Invalid fused op in binary_convolution kernel: " + params.layerID);
        }

        for (size_t op_input_id = 0; op_input_id < fused_dep.tensors.size(); op_input_id++) {
            std::string name = "FUSED_OP_" + std::to_string(op_id) + "_INPUT" + std::to_string(op_input_id);
            jit.AddConstant(MakeJitConstant(name, fused_dep.tensors[op_input_id]));
            input_decls += "const __global " + toCLType(fused_dep.tensors[op_input_id].GetDType()) + "* " + op_type +
                           "_input" + std::to_string(op_input_id) + ",";
        }

        if (fused_dep.activation.function != ActivationFunction::NONE) {
            std::string temp_op_type = op_type;
            for (auto& ch : temp_op_type) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
            std::string suffix = "_" + temp_op_type;

            jit.Merge(MakeActivationJitConstants(fused_dep.activation, suffix));
            eltwise_fused_ops += "res = ACTIVATION" + suffix + "(res, ACTIVATION_PARAMS" + suffix + ");";
        }
        op_id++;
    }
    jit.AddConstant(MakeJitConstant("FUSED_OPS_DECLS", input_decls));
    jit.AddConstant(MakeJitConstant("DO_ELTWISE_FUSED_OPS", eltwise_fused_ops));

    return jit;
}
}  // namespace kernel_selector
