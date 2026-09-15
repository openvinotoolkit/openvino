// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_elements_update_kernel_opt_local_sum.h"

#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {
// Mirrors _ref.cpp's file-local GetScatterElementsUpdateChannelIndex.
size_t GetChannelIndex(const scatter_elements_update_params& params) {
    const size_t input_size = params.inputs[0].GetDims().size();
    switch (params.axis) {
    case ScatterUpdateAxis::X:
        return input_size - 1;
    case ScatterUpdateAxis::Y:
        return input_size - 2;
    case ScatterUpdateAxis::Z:
        return input_size - 3;
    case ScatterUpdateAxis::W:
        return 2;
    case ScatterUpdateAxis::FEATURE:
        return 1;
    case ScatterUpdateAxis::BATCH:
        return 0;
    default:
        break;
    }
    return DataTensor::Channelndex(params.outputs[0].GetLayout(), Tensor::DataChannelName::X);
}
}  // namespace

ParamsKey ScatterElementsUpdateKernelOptLocalSum::GetSupportedKey() const {
    ParamsKey k;
    const std::vector<Datatype> supportedTypes{Datatype::F16,
                                               Datatype::F32,
                                               Datatype::INT32,
                                               Datatype::INT8,
                                               Datatype::UINT8};
    for (const auto t : supportedTypes) {
        k.EnableInputDataType(t);
        k.EnableOutputDataType(t);
    }

    // 4D and 5D only -- the .cl body indexes b/f/(z)/y/x explicitly.
    const std::vector<DataLayout> supportedLayots{DataLayout::bfyx,
                                                  DataLayout::b_fs_yx_fsv16,
                                                  DataLayout::b_fs_yx_fsv32,
                                                  DataLayout::bs_fs_yx_bsv16_fsv16,
                                                  DataLayout::bs_fs_yx_bsv32_fsv16,
                                                  DataLayout::bs_fs_yx_bsv16_fsv32,
                                                  DataLayout::bs_fs_yx_bsv32_fsv32,
                                                  DataLayout::bfzyx,
                                                  DataLayout::b_fs_zyx_fsv16,
                                                  DataLayout::b_fs_zyx_fsv32,
                                                  DataLayout::bs_fs_zyx_bsv16_fsv32,
                                                  DataLayout::bs_fs_zyx_bsv16_fsv16,
                                                  DataLayout::bs_fs_zyx_bsv32_fsv32,
                                                  DataLayout::bs_fs_zyx_bsv32_fsv16};
    for (const auto l : supportedLayots) {
        k.EnableInputLayout(l);
        k.EnableOutputLayout(l);
    }

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

KernelsPriority ScatterElementsUpdateKernelOptLocalSum::GetKernelsPriority(const Params& /*params*/) const {
    // `_ref` uses the base default, so force priority to win when both are eligible.
    return FORCE_PRIORITY_8;
}

CommonDispatchData ScatterElementsUpdateKernelOptLocalSum::SetDefault(const scatter_elements_update_params& params,
                                                                      bool is_second) const {
    CommonDispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    const auto& output = params.outputs[0];
    const auto& indices = params.inputs[1];
    const auto& scope = is_second ? indices : output;
    const auto rank = params.inputs[0].GetDims().size();

    // As in `_ref`: the update stage merges X*Y into gws[0], which the ITER == 1 body's
    // index decoding depends on. Init/finalize use the non-merged layout.
    if (is_second) {
        switch (rank) {
        case 4:
            dispatchData.gws = {indices.X().v * indices.Y().v, indices.Feature().v, indices.Batch().v};
            dims_by_gws = {{Tensor::DataChannelName::X, Tensor::DataChannelName::Y},
                           {Tensor::DataChannelName::FEATURE},
                           {Tensor::DataChannelName::BATCH}};
            break;
        case 5:
            dispatchData.gws = {indices.X().v * indices.Y().v, indices.Z().v * indices.Feature().v, indices.Batch().v};
            dims_by_gws = {{Tensor::DataChannelName::X, Tensor::DataChannelName::Y},
                           {Tensor::DataChannelName::Z, Tensor::DataChannelName::FEATURE},
                           {Tensor::DataChannelName::BATCH}};
            break;
        default:
            throw std::invalid_argument("Unsupported rank for scatter_elements_update_opt_local_sum");
        }
        dispatchData.lws =
            GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);
        return dispatchData;
    }

    switch (rank) {
    case 4:
        dispatchData.gws = {scope.X().v, scope.Y().v, scope.Feature().v * scope.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::X},
                       {Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
        break;
    case 5:
        dispatchData.gws = {scope.X().v * scope.Y().v, scope.Z().v, scope.Feature().v * scope.Batch().v};
        dims_by_gws = {{Tensor::DataChannelName::X, Tensor::DataChannelName::Y},
                       {Tensor::DataChannelName::Z},
                       {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
        break;
    default:
        throw std::invalid_argument("Unsupported rank for scatter_elements_update_opt_local_sum");
    }
    dispatchData.lws =
        GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

JitConstants ScatterElementsUpdateKernelOptLocalSum::GetJitConstants(
    const scatter_elements_update_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    jit.AddConstant(MakeJitConstant("AXIS_VALUE", GetChannelIndex(params)));
    jit.AddConstant(MakeJitConstant("WINDOW_SIZE", kWindowSize));
    return jit;
}

bool ScatterElementsUpdateKernelOptLocalSum::Validate(const Params& p) const {
    if (p.GetType() != KernelType::SCATTER_ELEMENTS_UPDATE) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    const auto& params = static_cast<const scatter_elements_update_params&>(p);

    if (params.mode != ScatterUpdateReduction::SUM) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    // Not in GetSupportedKey(): GetParamsKey() folds every input dtype into one bitfield,
    // and the indices tensor is INT32, so narrowing the key there would match nothing.
    const auto is_verified_type = [](Datatype dt) {
        return dt == Datatype::F16 || dt == Datatype::F32 || dt == Datatype::INT32;
    };
    if (!is_verified_type(params.inputs[0].GetDType()) || !is_verified_type(params.inputs[2].GetDType()) ||
        !is_verified_type(params.outputs[0].GetDType())) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    if (!params.use_init_val) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    // The .cl body derives the updates coordinates from the output rank.
    if (params.inputs[2].GetDims().size() != params.outputs[0].GetDims().size()) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    if (!params.fused_ops.empty()) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);  // `_ref` handles fused cases
    }
    // Every condition above is shape-independent, which is what lets this kernel serve
    // shape-agnostic compilation too.
    return true;
}

size_t ScatterElementsUpdateKernelOptLocalSum::GetAccumulatorSize(const DataTensor& output) {
    // One int32 slot per padded output element, plus a window of slack so a window anchored
    // near the end cannot run past it and the write-back needs no bounds check.
    return (output.PhysicalSize() + kWindowSize) * sizeof(int32_t);
}

void ScatterElementsUpdateKernelOptLocalSum::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const scatter_elements_update_params&>(params);
        OPENVINO_ASSERT(kd.kernels.size() == kShapeAgnosticKernelCount,
                        "[GPU] Invalid kernels size for scatter_elements_update_opt_local_sum");

        const auto& output = prim_params.outputs[0];
        const bool anchor_zero = AnchorAtZero(output);

        kd.internalBuffers.clear();
        kd.internalBuffers.push_back(GetAccumulatorSize(output));
        kd.internalBufferDataType = Datatype::INT32;

        for (size_t i = 0; i < kd.kernels.size(); ++i) {
            const bool is_update = (i == kStageUpdateAnchored || i == kStageUpdateFromZero);
            auto dispatchData = SetDefault(prim_params, /*is_second=*/is_update);
            kd.kernels[i].params.workGroups.global = dispatchData.gws;
            kd.kernels[i].params.workGroups.local = dispatchData.lws;
            // Exactly one of the two update variants runs, chosen by the shape just resolved.
            kd.kernels[i].skip_execution =
                KernelData::SkipKernelExecution(prim_params) ||
                (i == kStageUpdateFromZero && !anchor_zero) || (i == kStageUpdateAnchored && anchor_zero);
            if (is_update) {
                kd.kernels[i].params.local_memory_args.clear();
                kd.kernels[i].params.local_memory_args.push_back(kWindowSize * sizeof(int32_t));
            }
        }
    };
}

KernelsData ScatterElementsUpdateKernelOptLocalSum::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const auto& orgParams = static_cast<const scatter_elements_update_params&>(params);
    // Static: init, update, finalize. Shape-agnostic adds the second anchor variant.
    const bool shape_agnostic = orgParams.is_shape_agnostic;
    const size_t kernel_size = shape_agnostic ? kShapeAgnosticKernelCount : 3;

    KernelData kd = KernelData::Default<scatter_elements_update_params>(params, kernel_size);
    scatter_elements_update_params& newParams = *static_cast<scatter_elements_update_params*>(kd.params.get());
    auto cldnn_jit = GetJitConstants(newParams);

    if (shape_agnostic) {
        GetUpdateDispatchDataFunc(kd);
    }

    const auto& output = newParams.outputs[0];

    kd.internalBuffers.clear();
    kd.internalBuffers.push_back(GetAccumulatorSize(output));
    kd.internalBufferDataType = Datatype::INT32;

    for (size_t i = 0; i < kernel_size; i++) {
        const bool is_update =
            shape_agnostic ? (i == kStageUpdateAnchored || i == kStageUpdateFromZero) : (i == 1);
        const int32_t iter = is_update ? 1 : (i == 0 ? 0 : 2);
        const bool anchor_zero = shape_agnostic ? (i == kStageUpdateFromZero) : AnchorAtZero(output);

        auto dispatchData = SetDefault(newParams, /*is_second=*/is_update);
        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, i);
        clKernelData& kernel = kd.kernels[i];

        cldnn_jit.RemoveConstant("ITER");
        cldnn_jit.AddConstant(MakeJitConstant("ITER", iter));
        cldnn_jit.RemoveConstant("WINDOW_ANCHOR_ZERO");
        cldnn_jit.AddConstant(MakeJitConstant("WINDOW_ANCHOR_ZERO", anchor_zero ? 1 : 0));

        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        FillCLKernelData(kernel,
                         dispatchData,
                         params.engineInfo,
                         kernelName,
                         jit,
                         entry_point,
                         "",
                         false,
                         false,
                         3,
                         GetFusedPrimitiveInputsCount(params),
                         1,
                         shape_agnostic);

        // internal fixed-point accumulator buffer, every stage touches it
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});

        if (is_update) {
            // local staging window for the update stage only
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 0});
            kernel.params.local_memory_args.push_back(kWindowSize * sizeof(int32_t));
        }
    }

    return {kd};
}

}  // namespace kernel_selector
