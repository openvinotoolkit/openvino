// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_kernel_vload8.h"

#include <algorithm>
#include <string>

#include "activation/activation_kernel_base.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {

bool IsFeatureBroadcast(const DataTensor& input, const DataTensor& output) {
    if (input.GetLayout() != DataLayout::bfyx || output.GetLayout() != DataLayout::bfyx)
        return false;

    if (input.PitchesDifferFromLogicalDims() || output.PitchesDifferFromLogicalDims() || input.GetFirstElementOffset() != 0 ||
        output.GetFirstElementOffset() != 0)
        return false;

    return input.Batch().v == output.Batch().v && input.Feature().v == 1 && output.Feature().v > 1 && input.Y().v == output.Y().v &&
           input.X().v == output.X().v && (input.Y().v * input.X().v) % 8 == 0;
}

}  // namespace

ParamsKey EltwiseKernel_vload8::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::BF16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::BF16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableBatching();
    k.EnableEltwiseBroadcast();
    return k;
}

JitConstants EltwiseKernel_vload8::GetJitConstants(const eltwise_params& params) const {
    auto jit = GetJitConstantsCommon(params, true);

    const auto& output = params.outputs[0];
    const bool has_feature_broadcast = std::any_of(params.inputs.begin(), params.inputs.end(), [&](const DataTensor& input) {
        return IsFeatureBroadcast(input, output);
    });
    if (!has_feature_broadcast)
        return jit;

    const size_t feature_plane_vecs = output.Y().v * output.X().v / 8;
    const size_t output_batch_stride_vecs = output.Feature().v * feature_plane_vecs;
    std::string vload_decls;

    for (size_t i = 0; i < params.inputs.size(); i++) {
        const auto& input = params.inputs[i];
        vload_decls += "\\\n\tconst " + toCLType(input.GetDType()) + "8 in" + toCodeString(i);
        if (input.PhysicalSize() == 1) {
            vload_decls += " = (" + toCLType(input.GetDType()) + "8)(input" + toCodeString(i) + "[0]";
        } else if (IsFeatureBroadcast(input, output)) {
            // The output is flattened as [batch][feature][y*x]. Ignore the output
            // feature coordinate and reuse the single input feature plane.
            const std::string broadcast_idx = "((global_id / " + toCodeString(output_batch_stride_vecs) + ") * " + toCodeString(feature_plane_vecs) +
                                              " + (global_id % " + toCodeString(feature_plane_vecs) + "))";
            vload_decls += " = vload8(" + broadcast_idx + ", input" + toCodeString(i);
        } else {
            vload_decls += " = vload8(global_id, input" + toCodeString(i);
        }
        vload_decls += ");";
    }

    jit.RemoveConstant("VLOAD_DECLS");
    jit.AddConstant(MakeJitConstant("VLOAD_DECLS", vload_decls));
    return jit;
}

bool EltwiseKernel_vload8::Validate(const Params& params) const {
    if (!EltwiseKernelBase::Validate(params)) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    const auto& ewParams = static_cast<const eltwise_params&>(params);

    // Only one activation can be fused.
    if (ewParams.fused_ops.size() > 1 ||
        (!ewParams.activations.empty() && !ewParams.fused_ops.empty())) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

        for (size_t i = 0; i < ewParams.inputs.size(); i++) {
            const auto input_layout = ewParams.inputs[i].GetLayout();
            const auto batch_size = ewParams.inputs[i].Batch().v;
            const auto feature_size = ewParams.inputs[i].Feature().v;
            if ((input_layout == DataLayout::b_fs_yx_fsv16 && feature_size % 16 != 0) ||
                (input_layout == DataLayout::b_fs_yx_fsv32 && feature_size % 32 != 0) ||
                (input_layout == DataLayout::b_fs_zyx_fsv16 && feature_size % 16 != 0) ||
                (input_layout == DataLayout::b_fs_yx_fsv4 && feature_size % 8 != 0) ||
                input_layout == DataLayout::fs_b_yx_fsv32 ||
                (input_layout == DataLayout::bs_fs_yx_bsv32_fsv16 && (feature_size % 16 != 0 || batch_size % 32 != 0)) ||
                (input_layout == DataLayout::bs_fs_yx_bsv32_fsv32 && (feature_size % 32 != 0 || batch_size % 32 != 0))) {
                DO_NOT_USE_THIS_KERNEL(params.layerID);
            }
        }
        if ((ewParams.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv16 && ewParams.outputs[0].Feature().v % 16 != 0) ||
            (ewParams.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv32 && ewParams.outputs[0].Feature().v % 32 != 0) ||
            (ewParams.outputs[0].GetLayout() == DataLayout::b_fs_zyx_fsv16 && ewParams.outputs[0].Feature().v % 16 != 0) ||
            (ewParams.outputs[0].GetLayout() == DataLayout::b_fs_yx_fsv4 && ewParams.outputs[0].Feature().v % 8 != 0) ||
            ewParams.outputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32 ||
            (ewParams.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv16 &&
                (ewParams.outputs[0].Feature().v % 16 != 0 || ewParams.outputs[0].Batch().v % 32 != 0)) ||
            (ewParams.outputs[0].GetLayout() == DataLayout::bs_fs_yx_bsv32_fsv32 &&
                (ewParams.outputs[0].Feature().v % 32 != 0 || ewParams.outputs[0].Batch().v % 32 != 0))) {
            DO_NOT_USE_THIS_KERNEL(params.layerID);
        }

    const auto& output = ewParams.outputs[0];
    const auto count = output.PhysicalSize();

    const bool bSupportedCount = (count % 8) == 0;

    bool bCheckSizes = !output.PitchesDifferFromLogicalDims() && output.GetFirstElementOffset() == 0;
    for (size_t i = 0; i < ewParams.inputs.size(); i++) {
        // Allow equal-sized inputs, scalars, or a plain bfyx input which differs
        // only by broadcasting feature=1 to the output feature count.
        const bool same_as_output = ewParams.inputs[i] == output;
        const bool scalar = ewParams.inputs[i].PhysicalSize() == 1;
        const bool feature_broadcast = !ewParams.is_shape_agnostic && IsFeatureBroadcast(ewParams.inputs[i], output);
        if ((!same_as_output && !scalar && !feature_broadcast) || ewParams.inputs[i].PitchesDifferFromLogicalDims() ||
            ewParams.inputs[i].GetFirstElementOffset() != 0) {
            bCheckSizes = false;
        }
    }

    // TODO: add support to this implementation when user requests input values updates
    bool bCheckUpdateInput = true;
    if (!ewParams.updateInputIds.empty()) {
        bCheckUpdateInput = false;
    }

    // TODO: add support for reading from output buffer and using its values in computation
    bool bCheckUseOutput = true;
    for (size_t op = 0; op < ewParams.operations.size(); op++) {
        for (size_t input_idx = 0; input_idx < ewParams.operations[op].inputs.size(); input_idx++) {
            if (ewParams.operations[op].inputs[input_idx].mode == EltwiseInputMode::OUTPUT_BUFFER) {
                bCheckUseOutput = false;
                break;
            }
        }
    }

    if (IsUnsupportedModeForVecCode(ewParams)) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    if (!bCheckSizes || !bSupportedCount || !bCheckUpdateInput || !bCheckUseOutput) {
        DO_NOT_USE_THIS_KERNEL(params.layerID);
    }

    return true;
}

KernelsData EltwiseKernel_vload8::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<eltwise_params>(params);
    eltwise_params& newParams = *static_cast<eltwise_params*>(kd.params.get());

    std::pair<std::string, std::string> jit;

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);

    try {
        // move a fused activation from fused_ops to activations
        if (newParams.activations.empty() &&
            newParams.fused_ops.size() == 1 &&
            newParams.fused_ops[0].GetType() == KernelType::ACTIVATION) {
            auto p = newParams.fused_ops[0].GetOpParams<activation_fuse_params>();
            base_activation_params activation_p = p->param;
            newParams.activations.push_back(activation_p);
            newParams.fused_ops.clear();
        }

        auto cldnn_jit = GetJitConstants(newParams);
        jit = CreateJit(kernelName, cldnn_jit, entry_point);
    } catch (const std::runtime_error&) {
        return KernelsData();
    }

    auto& kernel = kd.kernels[0];
    kernel.params.workGroups.global = {std::max(newParams.outputs[0].LogicalSize() / 8, static_cast<size_t>(1)), 1, 1};
    kernel.params.workGroups.local = GetOptimalLocalWorkGroupSizes(kernel.params.workGroups.global, params.engineInfo);
    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, EXE_MODE_DEFAULT);
    kernel.params.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(), false, false);

    return {kd};
}

KernelsPriority EltwiseKernel_vload8::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}
}  // namespace kernel_selector
