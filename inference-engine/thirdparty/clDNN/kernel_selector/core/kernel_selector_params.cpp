// Copyright (c) 2016-2019 Intel Corporation
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


#include "kernel_selector_params.h"
#include "kernel_selector_common.h"
#include <sstream>
#include <string>
#include "jitter.h"

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ParamsKey
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ParamsKey::EnableInputDataType(Datatype dt) {
    switch (dt) {
        case Datatype::INT8:
            key.inputType.val.int8 = 1;
            break;
        case Datatype::UINT8:
            key.inputType.val.uint8 = 1;
            break;
        case Datatype::INT16:
            key.inputType.val.int16 = 1;
            break;
        case Datatype::UINT16:
            key.inputType.val.uint16 = 1;
            break;
        case Datatype::INT32:
            key.inputType.val.int32 = 1;
            break;
        case Datatype::UINT32:
            key.inputType.val.uint32 = 1;
            break;
        case Datatype::INT64:
            key.inputType.val.int64 = 1;
            break;
        case Datatype::F16:
            key.inputType.val.F16 = 1;
            break;
        case Datatype::F32:
            key.inputType.val.F32 = 1;
            break;
        case Datatype::BINARY:
            key.inputType.val.binary = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableAllInputDataType() { key.inputType.raw = 0xffffffff; }

void ParamsKey::EnableOutputDataType(Datatype dt) {
    switch (dt) {
        case Datatype::INT8:
            key.outputType.val.int8 = 1;
            break;
        case Datatype::UINT8:
            key.outputType.val.uint8 = 1;
            break;
        case Datatype::INT16:
            key.outputType.val.int16 = 1;
            break;
        case Datatype::UINT16:
            key.outputType.val.uint16 = 1;
            break;
        case Datatype::INT32:
            key.outputType.val.int32 = 1;
            break;
        case Datatype::UINT32:
            key.outputType.val.uint32 = 1;
            break;
        case Datatype::INT64:
            key.outputType.val.int64 = 1;
            break;
        case Datatype::F16:
            key.outputType.val.F16 = 1;
            break;
        case Datatype::F32:
            key.outputType.val.F32 = 1;
            break;
        case Datatype::BINARY:
            key.outputType.val.binary = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableAllOutputDataType() { key.outputType.raw = 0xffffffff; }

void ParamsKey::EnableInputWeightsType(WeightsType wt) {
    switch (wt) {
        case WeightsType::F16:
            key.inputWeightsType.val.F16 = 1;
            break;
        case WeightsType::F32:
            key.inputWeightsType.val.F32 = 1;
            break;
        case WeightsType::INT8:
            key.inputWeightsType.val.int8 = 1;
            break;
        case WeightsType::BINARY:
            key.inputWeightsType.val.binary = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableAllInputWeightsType() { key.inputWeightsType.raw = 0xffffffff; }

void ParamsKey::EnableOutputWeightsType(WeightsType wt) {
    switch (wt) {
        case WeightsType::F16:
            key.outputWeightsType.val.F16 = 1;
            break;
        case WeightsType::F32:
            key.outputWeightsType.val.F32 = 1;
            break;
        case WeightsType::INT8:
            key.outputWeightsType.val.int8 = 1;
            break;
        case WeightsType::BINARY:
            key.outputWeightsType.val.binary = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableAllOutputWeightsType() { key.outputWeightsType.raw = 0xffffffff; }

void ParamsKey::EnableLRNMode(LRNMode m) {
    switch (m) {
        case LRNMode::ACROSS_CHANNEL:
            key.restrict.val.dedicated.norm.across = 1;
            break;
        case LRNMode::WITHIN_CHANNEL:
            key.restrict.val.dedicated.norm.within = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableLookUpTableAxis(LookUpTableAxis m) {
    switch (m) {
        case kernel_selector::LookUpTableAxis::BATCH:
            key.restrict.val.dedicated.lookt.axisBatch = 1;
            break;
        case kernel_selector::LookUpTableAxis::FEATURE:
            key.restrict.val.dedicated.lookt.axisFeature = 1;
            break;
        case kernel_selector::LookUpTableAxis::X:
            key.restrict.val.dedicated.lookt.axisX = 1;
            break;
        case kernel_selector::LookUpTableAxis::Y:
            key.restrict.val.dedicated.lookt.axisY = 1;
            break;
        case kernel_selector::LookUpTableAxis::XYF:
            key.restrict.val.dedicated.lookt.axisXYF = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableNormalizeMode(NormalizeMode m) {
    switch (m) {
        case NormalizeMode::ACROSS_SPATIAL:
            key.restrict.val.dedicated.norm.across = 1;
            break;
        case NormalizeMode::WITHIN_SPATIAL:
            key.restrict.val.dedicated.norm.within = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableMVNMode(MVNMode m) {
    switch (m) {
        case MVNMode::ACROSS_CHANNELS:
            key.restrict.val.dedicated.mvn.across = 1;
            break;
        case MVNMode::WITHIN_CHANNELS:
            key.restrict.val.dedicated.mvn.within = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableMVNNormalizeVariance() { key.restrict.val.dedicated.mvn.normalize_variance = 1; }

void ParamsKey::EnableLRNKernelDividerMode(KernelDividerMode m) {
    switch (m) {
        case KernelDividerMode::FIXED:
            key.restrict.val.dedicated.norm.fixedKenrelDivider = 1;
            break;
        case KernelDividerMode::DYNAMIC:
            key.restrict.val.dedicated.norm.dynamicKenrelDivider = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnablePoolKernelDividerMode(KernelDividerMode m) {
    switch (m) {
        case KernelDividerMode::FIXED:
            key.restrict.val.dedicated.pooling.fixedKenrelDivider = 1;
            break;
        case KernelDividerMode::DYNAMIC:
            key.restrict.val.dedicated.pooling.dynamicKenrelDivider = 1;
            break;
        case KernelDividerMode::DYNAMIC_WITH_PADDING:
            key.restrict.val.dedicated.pooling.dynamicKenrelDividerWithPadding = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnablePoolType(PoolType t) {
    switch (t) {
        case PoolType::MAX:
            key.restrict.val.dedicated.pooling.max = 1;
            break;
        case PoolType::AVG:
            key.restrict.val.dedicated.pooling.avg = 1;
            break;
        case PoolType::MAX_WITH_ARGMAX:
            key.restrict.val.dedicated.pooling.max_with_argmax = 1;
            break;
        case PoolType::BILINEAR:
            key.restrict.val.dedicated.pooling.bilinear = 1;
            break;
        case PoolType::DEFORMABLE_BILINEAR:
            key.restrict.val.dedicated.pooling.deformable_bilinear = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnablePoolRemainder(PoolRemainder r) {
    switch (r) {
        case PoolRemainder::FLOOR:
            key.restrict.val.dedicated.pooling.floor = 1;
            break;
        case PoolRemainder::CEIL:
            key.restrict.val.dedicated.pooling.ceil = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableSoftmaxDim(SoftmaxDim d) {
    switch (d) {
        case SoftmaxDim::X:
            key.restrict.val.dedicated.softmax.dimX = 1;
            break;
        case SoftmaxDim::Y:
            key.restrict.val.dedicated.softmax.dimY = 1;
            break;
        case SoftmaxDim::FEATURE:
            key.restrict.val.dedicated.softmax.dimFeature = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableConcatAxis(ConcatAxis a) {
    switch (a) {
        case ConcatAxis::X:
            key.restrict.val.dedicated.concat.axisX = 1;
            break;
        case ConcatAxis::Y:
            key.restrict.val.dedicated.concat.axisY = 1;
            break;
        case ConcatAxis::Z:
            key.restrict.val.dedicated.concat.axisZ = 1;
            break;
        case ConcatAxis::W:
            key.restrict.val.dedicated.concat.axisW = 1;
            break;
        case ConcatAxis::FEATURE:
            key.restrict.val.dedicated.concat.axisFeature = 1;
            break;
        case ConcatAxis::BATCH:
            key.restrict.val.dedicated.concat.axisBatch = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableUpSamplingSampleType(SampleType a) {
    switch (a) {
        case SampleType::NEAREST:
            key.restrict.val.dedicated.upsample.nearest = 1;
            break;
        case SampleType::BILINEAR:
            key.restrict.val.dedicated.upsample.bilinear = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableFusedConvEltwEltwiseStride() { key.restrict.val.dedicated.fused_conv_eltw.stride = 1; }

void ParamsKey::EnableEltwiseStride() { key.restrict.val.dedicated.eltwise.stride = 1; }

void ParamsKey::EnableArgMaxMinAxis(ArgMaxMinAxis a) {
    switch (a) {
        case ArgMaxMinAxis::X:
            key.restrict.val.dedicated.argm.axisX = 1;
            break;
        case ArgMaxMinAxis::Y:
            key.restrict.val.dedicated.argm.axisY = 1;
            break;
        case ArgMaxMinAxis::Z:
            key.restrict.val.dedicated.argm.axisZ = 1;
            break;
        case ArgMaxMinAxis::FEATURE:
            key.restrict.val.dedicated.argm.axisFeature = 1;
            break;
        case ArgMaxMinAxis::BATCH:
            key.restrict.val.dedicated.argm.axisBatch = 1;
            break;
        case ArgMaxMinAxis::XYF:
            key.restrict.val.dedicated.argm.axisXYF = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableIndexSelectAxis(IndexSelectAxis a) {
    switch (a) {
        case IndexSelectAxis::X:
            key.restrict.val.dedicated.idxsel.axisX = 1;
            break;
        case IndexSelectAxis::Y:
            key.restrict.val.dedicated.idxsel.axisY = 1;
            break;
        case IndexSelectAxis::FEATURE:
            key.restrict.val.dedicated.idxsel.axisFeature = 1;
            break;
        case IndexSelectAxis::BATCH:
            key.restrict.val.dedicated.idxsel.axisBatch = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableLookUpTableIndicesFormat(Datatype a) {
    if (a == Datatype::F32)
        key.restrict.val.dedicated.lookt.indicesF32 = 1;
    else
        key.restrict.val.dedicated.lookt.indicesOther = 1;
}

void ParamsKey::EnableFusedConvEltwiseRWOutOpt() { key.restrict.val.dedicated.fused_conv_eltw.rw_out_opt = 1; }

bool ParamsKey::Support(const ParamsKey& k) const {
    if (!((key.restrict.raw & k.key.restrict.raw) == k.key.restrict.raw))  // check if this kernel supports this params
        return false;
    if (!((key.machineInfo.raw & k.key.machineInfo.raw) ==
          key.machineInfo.raw))  // check if machine supports this kernel
        return false;
    if (!((key.inputType.raw & k.key.inputType.raw) == k.key.inputType.raw))
        return false;
    if (!((key.outputType.raw & k.key.outputType.raw) == k.key.outputType.raw))
        return false;
    if (!((key.inputWeightsType.raw & k.key.inputWeightsType.raw) == k.key.inputWeightsType.raw))
        return false;
    if (!((key.outputWeightsType.raw & k.key.outputWeightsType.raw) == k.key.outputWeightsType.raw))
        return false;
    if (!((key.inputLayout & k.key.inputLayout) != 0 || key.inputLayout == k.key.inputLayout))
        return false;
    if (!((key.outputLayout & k.key.outputLayout) != 0 || key.outputLayout == k.key.outputLayout))
        return false;
    if (!((key.weightsInputLayout & k.key.weightsInputLayout) != 0 ||
          key.weightsInputLayout == k.key.weightsInputLayout))
        return false;
    if (!((key.weightsOutputLayout & k.key.weightsOutputLayout) != 0 ||
          key.weightsOutputLayout == k.key.weightsOutputLayout))
        return false;

    return true;
}

ParamsKey ParamsKey::Merge(const ParamsKey& k) const {
    ParamsKey ret;
    ret.key.restrict.raw = key.restrict.raw | k.key.restrict.raw;
    ret.key.machineInfo.raw = key.machineInfo.raw | k.key.machineInfo.raw;
    ret.key.inputType.raw = key.inputType.raw | k.key.inputType.raw;
    ret.key.outputType.raw = key.outputType.raw | k.key.outputType.raw;
    ret.key.inputWeightsType.raw = key.inputWeightsType.raw | k.key.inputWeightsType.raw;
    ret.key.outputWeightsType.raw = key.outputWeightsType.raw | k.key.outputWeightsType.raw;
    ret.key.inputLayout = key.inputLayout | k.key.inputLayout;
    ret.key.outputLayout = key.outputLayout | k.key.outputLayout;
    ret.key.weightsInputLayout = key.weightsInputLayout | k.key.weightsInputLayout;
    ret.key.weightsOutputLayout = key.weightsOutputLayout | k.key.weightsOutputLayout;
    return ret;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ParamsKey Params::GetParamsKey() const {
    ParamsKey k;

    if (engineInfo.bSubGroupSupport) {
        k.EnableSubGroup();
    }

    if (engineInfo.bSubGroupShortSupport) {
        k.EnableSubGroupShort();
    }

    return k;
}

std::string Params::to_string() const {
    std::stringstream s;
    s << toString(kType);
    return s.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ParamsKey optional_params::GetSupportedKey() const {
    ParamsKey k;

    for (auto l : inputLayouts) {
        k.EnableInputLayout(l);
    }

    for (auto l : outputLayouts) {
        k.EnableOutputLayout(l);
    }

    return k;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// base_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
ParamsKey base_params::GetParamsKey() const {
    ParamsKey k = Params::GetParamsKey();

    bool bBatching = false;
    bool bPitches = false;
    bool bOffests = false;
    bool bDifferentTypes = false;
    bool bFP16Used = (output.GetDType() == Datatype::F16);

    for (const auto& i : inputs) {
        k.EnableInputDataType(i.GetDType());
        k.EnableInputLayout(i.GetLayout());

        bBatching |= (i.Batch().v > 1);
        bPitches |= (i.PitchesDifferFromLogicalDims());
        bOffests |= (i.GetFirstElementOffset() != 0);
        bDifferentTypes |= (i.GetDType() != output.GetDType());
        bFP16Used |= (i.GetDType() == Datatype::F16);
    }

    k.EnableOutputDataType(output.GetDType());
    k.EnableOutputLayout(output.GetLayout());

    if (bBatching) {
        k.EnableBatching();
    }

    if (bPitches || output.PitchesDifferFromLogicalDims()) {
        k.EnableTensorPitches();
    }

    if (bDifferentTypes) {
        k.EnableDifferentTypes();
    }

    if (bOffests || output.GetFirstElementOffset() != 0) {
        k.EnableTensorOffset();
    }

    if (!engineInfo.bFP16Support && bFP16Used) {
        // I'm not sure it's the best idea, but we can live with it right now
        k.EnableFP16Emulation();
    }

    if (gradient) {
        k.EnableGradient();
    }

    return k;
}

std::string base_activation_params::to_string() const {
    std::stringstream s;
    s << "m" << m << "_n" << n << "_" << toString(function);
    return s.str();
}

std::string base_params::to_string() const {
    std::stringstream s;
    s << Params::to_string() << "_";
    // TODO: here should be loop through all fused Activations but it affects on result hash, which used by autoTune
    // option, that why only first Activation is used
    if (activations.size() >= 1)
        s << activations[0].to_string() << "_";
    if (activations.empty()) {
        s << "m" << 0.f << "_n" << 0.f << "_" << toString(ActivationFunction::NONE) << "_";
    }

    for (auto input : inputs) {
        s << toString(input) << "_";
    }
    s << toString(output);

    return s.str();
}


std::string base_params::fused_operation_desc::GetTypeStr() const {
    switch (type) {
        case Type::ELTWISE: return "eltwise";
        case Type::SCALE: return "scale";
        case Type::QUANTIZE: return "quantize";
        case Type::ACTIVATION: return "activation";
        case Type::UNDEFINED: return "";
        default: return "";
    }
}

JitConstants base_params::fused_operation_desc::MakeFusedTensorJitConstants(const FusedOpsConfiguration& /*conf*/) const {
    JitConstants jit{};
    for (size_t op_input_id = 0; op_input_id < tensors.size(); op_input_id++) {
        std::string name = GetInputTensorName(op_input_id);
        jit.AddConstant(MakeJitConstant(name, tensors[op_input_id]));
    }
    jit.AddConstant(MakeJitConstant(GetOutputTensorName(), output_tensor));
    return jit;
}

JitConstants base_params::fused_operation_desc::MakeInputDeclsJitConstants(const FusedOpsConfiguration& /*conf*/) const {
    JitConstants jit = {};

    std::string input_decls = "";
    for (size_t op_input_id = 0; op_input_id < tensors.size(); op_input_id++) {
        std::string ptr_name = GetInputPtrName(op_input_id);
        input_decls += "\\\n\tconst __global " + toCLType(tensors[op_input_id].GetDType()) +
                       "* " + ptr_name + (op_input_id == tensors.size() - 1 ? "" : ",");
    }

    jit.AddConstant(MakeJitConstant("FUSED_OP"+std::to_string(op_id)+"_DECLS", input_decls));
    return jit;
}

JitConstants base_params::fused_operation_desc::MakeLoadJitConstants(const FusedOpsConfiguration& conf) const {
    JitConstants jit = {};

    auto vec_size = conf.vec_size;
    auto idx = conf.bfyx_idx_order;

    std::string load_decls = "";
    static int i = 0;
    bool reuse_index = type == Type::QUANTIZE;
    std::string reused_idx = "reused_idx_" + std::to_string(i++);
    if (reuse_index) {
        load_decls += "\\\n\tint " + reused_idx + " = " +  GetIdx(0, idx_desc{idx}, conf.safe_load) + ";";
    }

    for (size_t op_input_id = 0; op_input_id < tensors.size(); op_input_id++) {
        if (type == Type::QUANTIZE && tensors.size() > 4 &&(op_input_id == 2 || op_input_id == 3))
            continue;
        load_decls += "\\\n\t" + GetInputTypeName(op_input_id, vec_size) + " " + GetInputVarName(op_input_id) + " = " +
                      GetJitLoad(conf, op_input_id, reuse_index, reused_idx) + ";";
    }
    jit.AddConstant(MakeJitConstant("FUSED_OP"+std::to_string(op_id)+"_LOAD" + conf.suffix, load_decls));

    return jit;
}

JitConstants base_params::fused_operation_desc::MakeOpJitConstants(const FusedOpsConfiguration& conf,
                                                                   std::string in_var, std::string& out_var) const {
    JitConstants jit = {};

    std::string op_decls = "";
    auto typed_activation = conf.typed_activation;
    auto vec_size = conf.vec_size;
    auto idx = conf.bfyx_idx_order;

    out_var = GetOutputVarName(in_var);
    switch (type) {
        case Type::SCALE:
            op_decls += "\\\n\t" + GetOutputType(vec_size) + " " + out_var + " = " + GetInputVarName(0) + " * " + in_var + ";";
            if (tensors.size() > 1)
                op_decls += "\\\n\t" + out_var + " += " + GetInputVarName(1) + ";";
            break;
        case Type::ELTWISE:
            op_decls += "\\\n\t" + GetOutputType(vec_size) + " " + out_var + " = " + ConvertToOutputType(GetInputVarName(0), vec_size) +
                        " + " + ConvertToOutputType(in_var, vec_size) + ";";
            break;
        case Type::QUANTIZE:
            op_decls += "\\\n\t" + in_var + " = min(max(" + GetInputVarName(0) + ", " + in_var + "), " + GetInputVarName(1)+");";
            op_decls += "\\\n\t" + in_var + " = round(" + in_var + "*" + GetInputVarName(4) + " + " + GetInputVarName(5) + ");";
            op_decls += "\\\n\t" + in_var + " = " + in_var + "*" + GetInputVarName(6) + " + " + GetInputVarName(7) + ";";
            op_decls += "\\\n\t" + GetOutputType(vec_size) + " " + out_var + " = " + ConvertToOutputTypeSat(in_var, vec_size) +";";
            break;
        case Type::ACTIVATION:
            op_decls += "\\\n\t" + GetOutputType(vec_size) + " " + out_var + " = " + in_var + ";";
            // Do nothing since activation call will be added later
            break;
        default: break;
    }

    if (activation.function != ActivationFunction::NONE) {
        auto suffix = "_FUSED_OP"+std::to_string(op_id) + conf.suffix;
        if (tensors.size() == 1) {
            jit.Merge(JitConstants{MakeJitConstant("NL_M" + suffix, GetInputVarName(0)),
                                   MakeJitConstant("NL_N" + suffix, activation.n)});
        } else {
            jit.Merge(JitConstants{MakeJitConstant("NL_M" + suffix, activation.m),
                                   MakeJitConstant("NL_N" + suffix, activation.n)});
        }
        // Disable type casts in activation, since current jit generator for activation don't respect vector size of parameters.
        // So conversion is explicitly done in params declaration
        jit.Merge(MakeActivationJitConstants(activation.function, suffix, false, true));
        if (typed_activation) {
            std::string params = ConvertToOutputType("NL_M" + suffix, vec_size) + ","+ ConvertToOutputType("NL_N" + suffix, vec_size);
            op_decls += "\\\n\t" + out_var + " = ACTIVATION_FUNC" + suffix + "(" + out_var + ", " + params + ");";
        } else {
            op_decls += "\\\n\t" + out_var + " = ACTIVATION" + suffix + "(" + out_var + ", ACTIVATION_PARAMS" + suffix + ");";
        }
    }

    jit.AddConstant(MakeJitConstant("FUSED_OP"+std::to_string(op_id)+"_ACTION" + conf.suffix, op_decls));

    return jit;
}

std::string base_params::fused_operation_desc::GetInputTensorName(size_t input_id) const {
    return "FUSED_OP_" + std::to_string(op_id) + "_INPUT" + std::to_string(input_id);
}

std::string base_params::fused_operation_desc::GetOutputTensorName() const {
    return "FUSED_OP_" + std::to_string(op_id) + "_OUTPUT";
}

std::string base_params::fused_operation_desc::GetInputTypeName(size_t input_id, size_t vec_size) const {
    if (vec_size == 0 || vec_size > 8)
        throw std::invalid_argument("Invalid vector size in jit definitions: " + std::to_string(vec_size));
    std::string scalar_type = GetInputTensorName(input_id) + "_TYPE";
    if (vec_size > 1)
        return "MAKE_VECTOR_TYPE(" + scalar_type + "," + std::to_string(vec_size) + ")";
    else
        return scalar_type;
}

std::string base_params::fused_operation_desc::GetIdx(size_t input_id, idx_desc idx, bool should_be_safe) const {
    std::string idx_order = "";
    if (tensors[input_id].Batch().v == 1) {
        idx.b = "0";
    }
    if (tensors[input_id].Feature().v == 1) {
        idx.f = "0";
    }
    if (tensors[input_id].Y().v == 1) {
        idx.y = "0";
    }
    if (tensors[input_id].X().v == 1) {
        idx.x = "0";
    }
    if (idx.dims == 4) {
        idx_order = idx.b + "," + idx.f + "," + idx.y + "," + idx.x;
    } else if (idx.dims == 5) {
        idx_order = idx.b + "," + idx.f + "," + idx.z + "," + idx.y + "," + idx.x;
    }

    if (should_be_safe)
        return GetInputTensorName(input_id) + "_GET_INDEX_SAFE(" + idx_order +")";
    else
        return GetInputTensorName(input_id) + "_GET_INDEX(" + idx_order +")";
}

std::string base_params::fused_operation_desc::GetJitLoad(const FusedOpsConfiguration& conf, size_t input_id,
                                                          bool reuse_index, std::string reused_idx) const {
    auto vec_size = 1;
    // TODO: Need to check input tensors here to make sure that they have the same layout as output
    if (type == Type::ELTWISE) {
        vec_size = conf.vec_size;
    }

    auto aligned = conf.aligned_load;
    auto idx = conf.bfyx_idx_order;
    if (vec_size == 0 || vec_size > 8)
        throw std::invalid_argument("Invalid vector size in jit definitions: " + std::to_string(vec_size));

    std::string index_func_call_vec = reuse_index ? reused_idx : GetIdx(input_id, idx_desc{idx}, conf.safe_load);
    std::string index_func_call = reuse_index ? reused_idx : GetIdx(input_id, idx_desc{idx}, conf.safe_load);
    if (conf.simple_offset) {
        std::string offset = conf.bfyx_idx_order[0];
        if (conf.safe_load)
            offset = "(" + offset + " % " + std::to_string(tensors[input_id].LogicalSize()) + ")";
        if (vec_size > 1)
            return "((const __global " + toCLType(tensors[input_id].GetDType()) + std::to_string(vec_size) + "*)(" +
                   GetInputPtrName(input_id) + " + " + offset + "))[0]";
        else
            return GetInputPtrName(input_id) + "[" + offset + "]";
    } else {
        if (aligned) {
            if (vec_size > 1)
                return " UNIT_BLOCK_READ" + std::to_string(vec_size) + "(" + GetInputPtrName(input_id) + ", " + index_func_call_vec + ")";
            else
                return " UNIT_BLOCK_READ(" + GetInputPtrName(input_id) + ", " + index_func_call + ")";
        } else {
            if (vec_size > 1)
                return "((const __global " + toCLType(tensors[input_id].GetDType()) + std::to_string(vec_size) + "*)(" +
                       GetInputPtrName(input_id) + " + " + index_func_call_vec + "))[0]";
            else
                return GetInputPtrName(input_id) + "[" + index_func_call + "]";
        }
    }
}

std::string base_params::fused_operation_desc::GetInputPtrName(size_t input_id) const {
    return GetTypeStr() + std::to_string(op_id) + "_input" + std::to_string(input_id);
}

std::string base_params::fused_operation_desc::GetInputVarName(size_t input_id) const {
    return GetTypeStr() + std::to_string(op_id) + "_data" + std::to_string(input_id);
}

std::string base_params::fused_operation_desc::GetOutputVarName(std::string input_var) const {
    static int i = 0;
    std::replace(input_var.begin(), input_var.end(), '[', '_');
    std::replace(input_var.begin(), input_var.end(), ']', '_');
    std::replace(input_var.begin(), input_var.end(), ' ', '_');
    return input_var + "_" + std::to_string(i++);
}

std::string base_params::fused_operation_desc::GetOutputType(size_t vec_size) const {
    if (vec_size > 1)
        return toCLType(output_tensor.GetDType()) + std::to_string(vec_size);
    else
        return toCLType(output_tensor.GetDType());
}

std::string base_params::fused_operation_desc::ConvertToOutputType(std::string var, size_t vec_size) const {
    return "convert_" + GetOutputType(vec_size) + "(" + var + ")";
}

std::string base_params::fused_operation_desc::ConvertToOutputTypeSat(std::string var, size_t vec_size) const {
    if (output_tensor.GetDType() == Datatype::F32 || output_tensor.GetDType() == Datatype::F16)
        return "convert_" + GetOutputType(vec_size) + "(" + var + ")";
    else
        return "convert_" + GetOutputType(vec_size) + "_sat(" + var + ")";
}


}  // namespace kernel_selector
