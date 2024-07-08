// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "convolution_kernel_b_fs_yx_fsv16_1x1.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {

ConvolutionKernel_b_fs_yx_fsv16_1x1::ConvolutionKernel_b_fs_yx_fsv16_1x1() : ConvolutionKernelBase("convolution_gpu_bfyx_f16_1x1") {
    std::vector<size_t> outputBlockWidths = { 1, 2, 4, 8 };
    std::vector<std::string> executionModes = ConvolutionKernelBase::autoTuneOptions;

    for (auto w : outputBlockWidths) {
        for (auto exeMode : executionModes) {
            autoTuneOptions.emplace_back(AutoTuneOption{w, exeMode});
        }
    }
}

ConvolutionKernel_b_fs_yx_fsv16_1x1::AutoTuneOption ConvolutionKernel_b_fs_yx_fsv16_1x1::GetAutoTuneOptions(const convolution_params& params,
                                                                                                            int /*autoTuneIndex*/) const {
    if (!params.has_dynamic_outputs()) {
        auto x = params.outputs[0].X().v;
        auto y = params.outputs[0].Y().v;
        auto f = params.outputs[0].Feature().v;

        if (x == 1 && y == 1) {
            return { 1, EXE_MODE_DEFAULT };
        } else if (x * f <= 256) {
            if (x < 8 || x * f <= 128)
                return { 2, EXE_MODE_DEFAULT };
            else
                return { 4, EXE_MODE_DEFAULT };
        } else if (x * f <= 1536) {
            return { 4, EXE_MODE_DEFAULT };
        } else {
            return { 8, EXE_MODE_DEFAULT };
        }
    } else {
        // In shape agnostic kernel, the output shape can not be specified at build time,
        // So we prepare 4 kernels(blockWith 1, 2, 4, 8) in advance and then use proper kernel at runtime when static shape comes.
        return { 8, EXE_MODE_DEFAULT };
    }
}

float ConvolutionKernel_b_fs_yx_fsv16_1x1::EstimateOccupancy(const convolution_params& params,
                                                             const ConvolutionTuningData& tuning_data) const {
    auto tuneOptions = GetAutoTuneOptions(params, 0);
    auto blockWidth = tuneOptions.blockWidth;

    auto x = params.outputs[0].X().v;
    auto y = params.outputs[0].Y().v;
    auto f = params.outputs[0].Feature().v;
    auto b = params.outputs[0].Batch().v;

    auto threads = CeilDiv(x * y, blockWidth) * CeilDiv(f, tuning_data.feature_block_size) * tuning_data.slm_div_factor * b;

    return static_cast<float>(threads) / static_cast<float>(params.engineInfo.maxThreadsPerDevice);
}

ConvolutionKernel_b_fs_yx_fsv16_1x1::ConvolutionTuningData ConvolutionKernel_b_fs_yx_fsv16_1x1::GetTuningParams(const convolution_params& params) const {
    ConvolutionTuningData tuning_data;

    if (!params.is_shape_agnostic) {
        const auto& input = params.inputs[0];
        bool block_size_one_is_better = params.outputs[0].X().v == 1 && params.outputs[0].Y().v == 1 && input.Feature().v >= 2048;

        // Accuracy issue is found with input.Feature() > 16 in static kernel, Need to fix later.
        if (params.engineInfo.deviceType == dev_type::integrated_gpu && params.engineInfo.supports_imad && !block_size_one_is_better) {
            size_t ic_blocks = CeilDiv(input.Feature().v, tuning_data.feature_block_size);
            size_t max_slm_div_factor = params.engineInfo.maxWorkGroupSize / tuning_data.sub_group_size;

            while (ic_blocks % (tuning_data.slm_div_factor * 2) == 0 && (tuning_data.slm_div_factor * 2 <= max_slm_div_factor) &&
                EstimateOccupancy(params, tuning_data) < 4.0)
                tuning_data.slm_div_factor *= 2;
        }
    }

    tuning_data.work_group_size = tuning_data.slm_div_factor * tuning_data.sub_group_size;

    return tuning_data;
}

ParamsKey ConvolutionKernel_b_fs_yx_fsv16_1x1::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_b_fs_yx_fsv16_1x1::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_b_fs_yx_fsv16_1x1::SetDefault(const convolution_params& params,
                                                                               int autoTuneIndex) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(params);

    ConvolutionTuningData tuning_data = GetTuningParams(params);

    auto autoTune = GetAutoTuneOptions(params, autoTuneIndex);
    dispatchData.cldnnStyle.blockWidth = autoTune.blockWidth;

    const auto& out = params.outputs[0];

    auto x = out.X().v;
    auto y = out.Y().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    dispatchData.gws[0] = x == 1 && y == 1 ? 1 : CeilDiv(x * y, autoTune.blockWidth);
    dispatchData.gws[1] = Align(f, tuning_data.feature_block_size) * tuning_data.slm_div_factor;
    dispatchData.gws[2] = b;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = tuning_data.work_group_size;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_b_fs_yx_fsv16_1x1::GetKernelsPriority(const Params& params) const {
    const auto& p = static_cast<const convolution_params&>(params);
    if (!p.is_shape_agnostic) {
        auto autoTune = GetAutoTuneOptions(p, -1);

        const auto& input = p.inputs[0];
        const auto& out = p.outputs[0];

        auto bBlockSizeX = out.X().v % autoTune.blockWidth == 0;
        auto bBlockSizeXY = out.X().pad.Total() + out.Y().pad.Total() == 0;
        auto bInputPad = input.X().pad.Total() + input.Y().pad.Total() != 0;

        if (out.Batch().v == 1) {
            if ((bBlockSizeX || bBlockSizeXY) && !bInputPad) {
                return FORCE_PRIORITY_1;
            } else {
                return FORCE_PRIORITY_3;
            }
        } else {
            return FORCE_PRIORITY_7;
        }
    } else {
        return FORCE_PRIORITY_1;
    }
}

bool ConvolutionKernel_b_fs_yx_fsv16_1x1::Validate(const Params& p) const {
    if (!ConvolutionKernelBase::Validate(p)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);

    ConvolutionTuningData tuning_data = GetTuningParams(params);

    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    const bool bOutputSizes = (!input.X().is_dynamic && !output.X().is_dynamic && output.X().v != input.X().v) ||
                              (!input.Y().is_dynamic && !output.Y().is_dynamic && output.Y().v != input.Y().v) ||
                              (!output.Feature().is_dynamic && output.Feature().v % 16 != 0);
    const bool bFilterSize = params.filterSize.x != 1 || params.filterSize.y != 1;
    const bool bStride = params.stride.x != 1 || params.stride.y != 1;
    const bool bPadding = (!input.Feature().pad.is_dynamic && input.Feature().pad.before % tuning_data.feature_block_size != 0) ||
                          (!output.Feature().pad.is_dynamic && output.Feature().pad.before % tuning_data.feature_block_size != 0);

    if  (bOutputSizes || bFilterSize || bStride || bPadding) {
        return false;
    }

    return true;
}

JitConstants ConvolutionKernel_b_fs_yx_fsv16_1x1::GetJitConstants(const convolution_params& params,
                                                                  const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    ConvolutionTuningData tuning_data = GetTuningParams(params);

    auto blockWidth = dispatchData.cldnnStyle.blockWidth;
    if (!params.fused_ops.empty()) {
        auto input_dt = GetUnitType(params);
        FusedOpsConfiguration conf_vec = { "_VEC",
                                           {"b", "(feature_block * 16)", "y", "x"},
                                           "dst",
                                           input_dt,
                                           blockWidth,
                                           LoadType::LT_ALIGNED_READ,
                                           BoundaryCheck::ENABLED,
                                           IndexType::TENSOR_COORD,
                                           Tensor::DataChannelName::X };
        FusedOpsConfiguration conf_scalar1 = { "_SCALAR",
                                              {"b", "(feature_block * 16)", "yi", "xi"},
                                              "dst[i]",
                                              input_dt,
                                              1,
                                              LoadType::LT_ALIGNED_READ,
                                              BoundaryCheck::ENABLED,
                                              IndexType::TENSOR_COORD,
                                              Tensor::DataChannelName::X };
        FusedOpsConfiguration conf_scalar2 = { "_SCALAR_B1",
                                              {"b", "(feature_block * 16)", "0", "0"},
                                              "dst",
                                              input_dt,
                                              1,
                                              LoadType::LT_ALIGNED_READ,
                                              BoundaryCheck::ENABLED,
                                              IndexType::TENSOR_COORD,
                                              Tensor::DataChannelName::X };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf_vec, conf_scalar1, conf_scalar2 }));
    }

    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", blockWidth));
    jit.AddConstant(MakeJitConstant("SLM_DIV_FACTOR", tuning_data.slm_div_factor));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", tuning_data.sub_group_size));
    jit.AddConstant(MakeJitConstant("WORK_GROUP_SIZE", tuning_data.work_group_size));

    if (!params.has_dynamic_inputs()) {
        jit.AddConstant(MakeJitConstant("PADDED_INPUT", params.inputs[0].X().pad.Total() != 0));

        bool padded_output = params.outputs[0].X().pad.Total() != 0;
        bool non_unit_fused_op_spatial = false;

        // Set padded_output to true when fused inputs have paddings to have correct blocked loads
        for (auto& fused_op : params.fused_ops) {
            for (auto& t : fused_op.tensors) {
                if (t.PitchesDifferFromLogicalDims()) {
                    padded_output = true;
                }
                if ((t.X().v > 1) ||
                    (t.Y().v > 1) ||
                    (t.Z().v > 1) ||
                    (t.W().v > 1)) {
                    non_unit_fused_op_spatial = true;
                }
            }
        }

        jit.AddConstant(MakeJitConstant("PADDED_OUTPUT", padded_output));
        jit.AddConstant(MakeJitConstant("NON_UNIT_FUSED_OP_SPATIAL", non_unit_fused_op_spatial));

        jit.AddConstant(MakeJitConstant("IC_BLOCKS", CeilDiv(params.inputs[0].Feature().v, tuning_data.feature_block_size)));
        if (params.outputs[0].Feature().v % tuning_data.feature_block_size != 0) {
            jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", 1));
        }
        if (params.inputs[0].Feature().v % tuning_data.feature_block_size != 0) {
            jit.AddConstant(MakeJitConstant("INPUT_LEFTOVERS", 1));
        }
    } else {
        DimensionAccessHelperJit input0_dims(params.inputs[0]);
        DimensionAccessHelperJit input0_padded_dims(params.inputs[0], true);
        DimensionAccessHelperJit output_dims(params.outputs[0]);
        DimensionAccessHelperJit output_padded_dims(params.outputs[0], true);

        const auto padded_input = "(" + input0_padded_dims.x_pad().first + "+" + input0_padded_dims.x_pad().first + ") != 0";
        jit.AddConstant(MakeJitConstant("PADDED_INPUT", padded_input));

        const auto padded_output = "(" + output_padded_dims.x_pad().first + "+" + output_padded_dims.x_pad().first + ") != 0";
        jit.AddConstant(MakeJitConstant("PADDED_OUTPUT", padded_output));

        // In shape agnostic kernel, the fused shape cannot be specified at build time or run time.
        // Currently simply check whether fused_op is dynmaic. Need to further follow up like static behavior.
        bool non_unit_fused_op_spatial = false;
        for (auto& fused_op : params.fused_ops) {
            for (auto& t : fused_op.tensors) {
                if (t.is_dynamic()) {
                    non_unit_fused_op_spatial = true;
                    break;
                } else {
                    if ((t.X().v > 1) ||
                        (t.Y().v > 1) ||
                        (t.Z().v > 1) ||
                        (t.W().v > 1)) {
                        non_unit_fused_op_spatial = true;
                        break;
                    }
                }
            }
        }
        jit.AddConstant(MakeJitConstant("NON_UNIT_FUSED_OP_SPATIAL", non_unit_fused_op_spatial));

        const auto feature_block_size = std::to_string(tuning_data.feature_block_size);
        const auto ic_blocks = "(" + input0_dims.f() + "+" + feature_block_size + " - 1) / " + feature_block_size;
        jit.AddConstant(MakeJitConstant("IC_BLOCKS", ic_blocks));

        const auto output_leftover_num = "(" + output_dims.f() + "%" + feature_block_size + ")";
        const auto output_leftover = "(" + output_leftover_num + "!= 0)";
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", output_leftover));

        const auto input_leftover_num = "(" + input0_dims.f() + "%" + feature_block_size + ")";
        const auto input_leftover = "(" + input_leftover_num + "!= 0)";
        jit.AddConstant(MakeJitConstant("INPUT_LEFTOVERS", input_leftover));
    }

    return jit;
}

 KernelsData ConvolutionKernel_b_fs_yx_fsv16_1x1::GetKernelsData(const Params& params) const {
    size_t num_kernels = params.is_shape_agnostic ? 4 : 1;
    KernelData kd = KernelData::Default<convolution_params>(params, num_kernels);
    convolution_params& newParams = *static_cast<convolution_params*>(kd.params.get());

    if (!Validate(params)) {
        return {};
    }

    auto preferredWeightsLayout = GetPreferredWeightsLayout(newParams);
    bool succeed = UpdateWeightsParams(newParams,
                                       preferredWeightsLayout,
                                       kd.weightsReorderParams,
                                       GetSupportedKey(),
                                       newParams.groups,
                                       newParams.transposed);

    bool bSupportedWeightsLayout = newParams.weights.GetLayout() == preferredWeightsLayout;
    const bool bWeightsOK = bSupportedWeightsLayout || newParams.allowStaticInputReordering;

    if (!succeed || !bWeightsOK) {
        return {};
    }

    if (NeedPaddedInput()) {
        if (newParams.has_dynamic_inputs()) {
            if (!CheckConvolutionExplicitPaddings(newParams))
                return {};
        } else {
            kd.reorderInput = ConvolutionUpdateInputParams(newParams);

            if (kd.reorderInput && !newParams.allowInputReordering)
                return {};
        }
    }

    DispatchData dispatchData = SetDefault(newParams, -1);

    if (!params.is_shape_agnostic && !CheckWorkGroups(dispatchData)) {
        // Internal Error - wrong calculation of global/local work group sizes
        return {};
    }

    auto finalKernelName = GetKernelName(newParams);
    auto cldnnJit = GetJitConstants(newParams, dispatchData);
    for (size_t i = 0; i < num_kernels; i++) {
        if (params.is_shape_agnostic) {
            if (i == 0) {
                dispatchData.cldnnStyle.blockWidth = 1;
            } else if (i == 1) {
                dispatchData.cldnnStyle.blockWidth = 2;
            } else if (i == 2) {
                dispatchData.cldnnStyle.blockWidth = 4;
            } else if (i == 3) {
                dispatchData.cldnnStyle.blockWidth = 8;
            }
            cldnnJit = GetJitConstants(newParams, dispatchData);
        }
        auto entryPoint = GetEntryPoint(finalKernelName, newParams.layerID, params, i);
        auto jit = CreateJit(finalKernelName, cldnnJit, entryPoint);

        GetUpdateDispatchDataFunc(kd);

        auto& kernel = kd.kernels[i];
        FillCLKernelData(kernel,
                        dispatchData,
                        params.engineInfo,
                        finalKernelName,
                        jit,
                        entryPoint,
                        EXE_MODE_DEFAULT,
                        true,
                        !newParams.bias.empty(),
                        1, 0, 1,
                        newParams.is_shape_agnostic);

        if (newParams.deformable_mode) {
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
            if (newParams.deformable_mask_enabled)
                kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});
        }

        if (!newParams.weights_zero_points.empty())
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::WEIGHTS_ZERO_POINTS, 1});
        if (!newParams.activations_zero_points.empty())
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::ACTIVATIONS_ZERO_POINTS, 1});
        if (!newParams.compensation.empty())
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::COMPENSATION, 1});

        uint32_t fused_deps_total = 0;
        for (auto& fused_dep : newParams.fused_ops) {
            for (int i = 0; i < static_cast<int>(fused_dep.dep_size); i++) {
                kernel.params.arguments.push_back({ ArgumentDescriptor::Types::INPUT_OF_FUSED_PRIMITIVE, fused_deps_total });
                fused_deps_total++;
            }
        }
    }
    kd.autoTuneIndex = -1;

    return {kd};
 }

void ConvolutionKernel_b_fs_yx_fsv16_1x1::GetUpdateDispatchDataFunc(KernelData& kd) const {
    if (kd.kernels.size() == 1) {
        Parent::GetUpdateDispatchDataFunc(kd);
    } else {
        kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
            const auto& prim_params = static_cast<const convolution_params&>(params);
            auto dispatchData = SetDefault(prim_params);
            size_t execute_kernel_idx = 3;
            if (dispatchData.cldnnStyle.blockWidth == 1) {
                execute_kernel_idx = 0;
            } else if (dispatchData.cldnnStyle.blockWidth == 2) {
                execute_kernel_idx = 1;
            } else if (dispatchData.cldnnStyle.blockWidth == 4) {
                execute_kernel_idx = 2;
            }
            for (size_t i = 0; i < kd.kernels.size(); i++) {
                kd.kernels[i].params.workGroups.global = dispatchData.gws;
                kd.kernels[i].params.workGroups.local = dispatchData.lws;
                if (execute_kernel_idx == i) {
                    kd.kernels[i].skip_execution = KernelData::SkipKernelExecution(prim_params);
                } else {
                    kd.kernels[i].skip_execution = true;
                }
            }
        };
    }
}

}  // namespace kernel_selector
