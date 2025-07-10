// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_selector_params.h"
#include "kernel_selector_common.h"
#include <sstream>
#include <string>

#include <activation/activation_kernel_base.h>
#include "jitter.h"

namespace kernel_selector {

DeviceFeaturesKey EngineInfo::get_supported_device_features_key() const {
    DeviceFeaturesKey k;

    if (supports_intel_subgroups) {
        k.enable_subgroup_shuffle_relative();
    }

    // Note: sub-group extension emulation is an experimental thing and may produce incorrect results in some cases.
    // Several known issues are listed below:
    // 1. Kernels with subgroups may be implemented for specific sub-group size which is controlled by cl_intel_required_subgroup_size extension.
    //    If that extension is unsupported then such kernels may produce wrong result.
    // 2. Offset for sub-group block read/write functions in some cases includes get_sub_group_local_id() value which seems to be processed correctly
    //    by intel extension, but may produce wrong result for emulation path.
    // If you face such kind of issue, you may want to disable emulation by setting enable_sub_groups_emulation = false in set_params() method
    bool can_emulate_intel_subgroups = enable_sub_groups_emulation && supports_khr_subgroups && (CL_TARGET_OPENCL_VERSION >= 200);

    if (can_emulate_intel_subgroups || supports_intel_subgroups) {
        k.enable_subgroups();
        k.enable_subgroup_reduce();
        k.enable_subgroup_broadcast();
        k.enable_reqd_subgroup_size();
        k.enable_blocked_read_write();
        k.enable_subgroup_shuffle();
        k.enable_blocked_read_write_short();
        k.enable_blocked_read_write_char();
    }

    return k;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ParamsKey
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void ParamsKey::EnableInputDataType(Datatype dt) {
    switch (dt) {
        case Datatype::INT4:
            key.inputType.val.int4 = 1;
            break;
        case Datatype::UINT4:
            key.inputType.val.uint4 = 1;
            break;
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
        case Datatype::BF16:
            key.inputType.val.BF16 = 1;
            break;
        default:
            break;
    }
}

void ParamsKey::EnableAllInputDataType() { key.inputType.raw = 0xffffffff; }

void ParamsKey::EnableOutputDataType(Datatype dt) {
    switch (dt) {
        case Datatype::INT4:
            key.outputType.val.int4 = 1;
            break;
        case Datatype::UINT4:
            key.outputType.val.uint4 = 1;
            break;
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
        case Datatype::BF16:
            key.outputType.val.BF16 = 1;
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
        case WeightsType::INT4:
            key.inputWeightsType.val.int4 = 1;
            break;
        case WeightsType::UINT4:
            key.inputWeightsType.val.uint4 = 1;
            break;
        case WeightsType::INT32:
            key.inputWeightsType.val.int32 = 1;
            break;
        case WeightsType::BF16:
            key.inputWeightsType.val.BF16 = 1;
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
        case WeightsType::INT4:
            key.outputWeightsType.val.int4 = 1;
            break;
        case WeightsType::UINT4:
            key.outputWeightsType.val.uint4 = 1;
            break;
        case WeightsType::INT32:
            key.outputWeightsType.val.int32 = 1;
            break;
        case WeightsType::BF16:
            key.outputWeightsType.val.BF16 = 1;
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
        case SoftmaxDim::Z:
            key.restrict.val.dedicated.softmax.dimZ = 1;
            break;
        case SoftmaxDim::FEATURE:
            key.restrict.val.dedicated.softmax.dimFeature = 1;
            break;
        case SoftmaxDim::BATCH:
            key.restrict.val.dedicated.softmax.dimBatch = 1;
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

void ParamsKey::EnableReampleType(ResampleType a) {
    switch (a) {
        case ResampleType::NEAREST_NEIGHBOR:
            key.restrict.val.dedicated.resample.nearest_neighbor = 1;
            break;
        case ResampleType::CAFFE_BILINEAR_INTERP:
            key.restrict.val.dedicated.resample.caffe_bilinear_interp = 1;
            break;
        case ResampleType::BILINEAR_INTERP:
            key.restrict.val.dedicated.resample.bilinear_interp = 1;
            break;
        case ResampleType::CUBIC:
            key.restrict.val.dedicated.resample.cubic = 1;
            break;
        case ResampleType::LINEAR_ONNX:
            key.restrict.val.dedicated.resample.linear_onnx = 1;
            break;
        case ResampleType::BILINEAR_PILLOW:
            key.restrict.val.dedicated.resample.bilinear_pillow = 1;
            break;
        case ResampleType::BICUBIC_PILLOW:
            key.restrict.val.dedicated.resample.bicubic_pillow = 1;
            break;
        default:
            break;
    }
}

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
        default:
            break;
    }
}

void ParamsKey::EnableQuantization(QuantizationType q) {
    switch (q) {
        case QuantizationType::NONE:
            break;
        case QuantizationType::SYMMETRIC:
            key.restrict.val.sym_quantization = 1;
            break;
        case QuantizationType::ASYMMETRIC_DATA:
            key.restrict.val.asym_d_quantization = 1;
            break;
        case QuantizationType::ASYMMETRIC_WEIGHTS:
            key.restrict.val.asym_w_quantization = 1;
            break;
        case QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS:
            key.restrict.val.asym_d_quantization = 1;
            key.restrict.val.asym_w_quantization = 1;
            break;
        default:
            break;
    }
}

bool ParamsKey::Support(const ParamsKey& k) const {
    if (!((key.restrict.raw & k.key.restrict.raw) == k.key.restrict.raw))  // check if this kernel supports this params
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
    return ParamsKey();
}

std::string Params::to_string() const {
    std::stringstream s;
    s << toString(kType);
    return s.str();
}

std::string Params::to_cache_string_v2() const {
    return "";
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
    // TODO : multiple output support
    bool bFP16Used = (outputs[0].GetDType() == Datatype::F16);

    bool dynamic_shapes = false;

    for (const auto& i : inputs) {
        k.EnableInputDataType(i.GetDType());
        k.EnableInputLayout(i.GetLayout());

        bBatching |= (i.Batch().v > 1);
        bPitches |= (i.PitchesDifferFromLogicalDims());
        bOffests |= (i.GetFirstElementOffset() != 0);
        bDifferentTypes |= (i.GetDType() != outputs[0].GetDType());
        bFP16Used |= (i.GetDType() == Datatype::F16);
        dynamic_shapes |= i.is_dynamic();
    }

    for (const auto& fused_op : fused_ops) {
        for (const auto& tensor : fused_op.tensors) {
            dynamic_shapes |= tensor.is_dynamic();
        }
    }

    k.EnableOutputDataType(outputs[0].GetDType());
    k.EnableOutputLayout(outputs[0].GetLayout());

    dynamic_shapes |= outputs[0].is_dynamic();

    if (bBatching) {
        k.EnableBatching();
    }

    if (bPitches || outputs[0].PitchesDifferFromLogicalDims()) {
        k.EnableTensorPitches();
    }

    if (bDifferentTypes) {
        k.EnableDifferentTypes();
    }

    if (bOffests || outputs[0].GetFirstElementOffset() != 0) {
        k.EnableTensorOffset();
    }

    if (dynamic_shapes) {
        k.EnableDynamicShapesSupport();
    }

    if (!engineInfo.supports_fp16 && bFP16Used) {
        // I'm not sure it's the best idea, but we can live with it right now
        k.EnableFP16Emulation();
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

    // WA to reuse old tuning cache. Code below must be replace with the following line once new cache file is merged.
    // s << Params::to_string() << "_";
    auto type_string = toString(kType);
    s << type_string << "_";

    // TODO: Remove activation from the string and recollect cache file
    bool found_fused_activation = false;
    if (!activations.empty()) {
        s << activations[0].to_string() << "_";
        found_fused_activation = true;
    }

    if (activations.empty() && !fused_ops.empty()) {
        if (fused_ops[0].GetType() == KernelType::ACTIVATION) {
            auto activation_params = fused_ops[0].GetOpParams<activation_fuse_params>()->param;
            s << activation_params.to_string() << "_";
            found_fused_activation = true;
        }
    }

    if (!found_fused_activation) {
        s << "m" << 0.f << "_n" << 0.f << "_" << toString(ActivationFunction::NONE) << "_";
    }

    for (auto input : inputs) {
        s << toString(input) << "_";
    }
    s << toString(outputs[0]);

    return s.str();
}

std::string base_params::to_cache_string_v2() const {
    std::stringstream s;

    for (auto input : inputs) {
        s << toString_v2(input) << ";";
    }
    s << toString_v2(outputs[0]);

    return s.str();
}

}  // namespace kernel_selector
