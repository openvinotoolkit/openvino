// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_elements_update_kernel_opt_local_sum.h"

#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {
bool is_global_memory_case(const scatter_elements_update_params& params) {
    return (params.outputs[0].PhysicalSizeInBytes() * 4 > params.engineInfo.maxLocalMemSize);
}

// Mirrors _ref.cpp's file-local GetScatterElementsUpdateChannelIndex (duplicated, not
// shared -- see the header comment for why).
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
    // As broad as `_ref`'s own key -- this is only a pre-filter checked before
    // Validate() ever runs (see kernel_selector_base::GetAllImplementations); the real
    // narrowing lives in Validate(), not here.
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
                                                  DataLayout::bs_fs_zyx_bsv32_fsv16,
                                                  DataLayout::bfwzyx};
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
    // `_ref` uses the base default (DONT_USE_IF_HAVE_SOMETHING_ELSE); force priority so
    // this kernel wins when both are eligible, matching GridSample's opt-kernel precedent.
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

    // Matches `_ref`'s own SetDefault: the update stage merges X*Y into gws[0] as one
    // dispatch dimension, which the ITER==1 .cl body's `x = dim0 % INPUT2_SIZE_X; y =
    // dim0 / INPUT2_SIZE_X;` decoding depends on. Init/finalize use the non-merged layout.
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
    // Element budget of the internal accumulator buffer (matches GetKernelsData's
    // allocation) -- bounds the write-back loop so a window straddling the buffer's end
    // can't write out of bounds.
    jit.AddConstant(MakeJitConstant("OPT_LOCAL_ACC_TOTAL_ELEMENTS", params.outputs[0].PhysicalSize()));
    return jit;
}

bool ScatterElementsUpdateKernelOptLocalSum::Validate(const Params& p) const {
    if (p.GetType() != KernelType::SCATTER_ELEMENTS_UPDATE) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    const auto& params = static_cast<const scatter_elements_update_params&>(p);

    // Rejecting here falls through to `_ref` unchanged -- this kernel only opts in for
    // its narrow scope, never replaces `_ref` for anything outside it.
    if (params.mode != ScatterUpdateReduction::SUM) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    // Accept only the element types this kernel is actually exercised on. The encoding
    // below handles every type in GetSupportedKey() -- integers included, since `_ref`'s
    // identity branch is reproduced here and the accumulator is sized per element rather
    // than per output byte -- but i8/u8 cannot currently reach any scatter kernel as an
    // input (the plugin's impl gate allows only f32/f16/i32 there) and so cannot be
    // tested. Rejecting them keeps the accepted set equal to the verified set: if that
    // gate is ever relaxed, this kernel steps aside for `_ref` rather than quietly
    // taking a path nobody has run.
    //
    // This has to live here rather than in GetSupportedKey(): base_params::GetParamsKey()
    // folds every input's dtype into one shared bitfield, and the indices tensor is INT32,
    // so narrowing the key's type list would stop the kernel matching anything at all.
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
    if (params.is_shape_agnostic) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);  // static shapes only for this first version
    }
    if (!is_global_memory_case(params)) {
        // `_ref`'s own whole-output-fits-in-local-memory path already wins here
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    if (!params.fused_ops.empty()) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);  // keep the first version simple; `_ref` still handles fused cases
    }
    const size_t rank = params.inputs[0].GetDims().size();
    if (rank != 4 && rank != 5) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    // Dense-ish scatter only: sparse scatters (few indices into a huge output) would
    // pay for zeroing/flushing a window with no locality benefit.
    if (params.inputs[2].LogicalSize() < params.outputs[0].LogicalSize()) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }
    return true;
}

bool ScatterElementsUpdateKernelOptLocalSum::SkipKernelExecution(const scatter_elements_update_params& params,
                                                                 size_t kernel_id) const {
    if (kernel_id == 0) {
        if (params.outputs[0].LogicalSize() != 0 && params.outputs[0] != params.inputs[0]) {
            return false;
        }
    }
    return KernelData::SkipKernelExecution(params);
}

void ScatterElementsUpdateKernelOptLocalSum::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const scatter_elements_update_params&>(params);
        OPENVINO_ASSERT(kd.kernels.size() == 3, "[GPU] Invalid kernels size for scatter_elements_update_opt_local_sum");

        for (size_t i = 0; i < kd.kernels.size(); ++i) {
            // is_second==true selects the "update" (ITER==1) dispatch shape (sized to
            // the indices/updates tensor, which for us equals the output size anyway).
            auto dispatchData = SetDefault(prim_params, /*is_second=*/i == 1);
            kd.kernels[i].params.workGroups.global = dispatchData.gws;
            kd.kernels[i].params.workGroups.local = dispatchData.lws;
            kd.kernels[i].skip_execution = SkipKernelExecution(prim_params, i);

            if (i == 1) {
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

    const size_t kernel_size = 3;  // STAGE0 init, STAGE1 update (local-staged), STAGE2 finalize

    KernelData kd = KernelData::Default<scatter_elements_update_params>(params, kernel_size);
    scatter_elements_update_params& newParams = *static_cast<scatter_elements_update_params*>(kd.params.get());
    auto cldnn_jit = GetJitConstants(newParams);

    GetUpdateDispatchDataFunc(kd);

    const auto& output = newParams.outputs[0];

    kd.internalBuffers.clear();
    // One int32 accumulator slot per (padded) output element. Sized from the element
    // count, not from the output's byte size: `_ref` writes this same buffer as
    // `PhysicalSizeInBytes() * 2`, which happens to equal one int32 per element only for
    // 2-byte types and under-allocates by half for i8/u8.
    kd.internalBuffers.push_back(output.PhysicalSize() * sizeof(int32_t));
    kd.internalBufferDataType = Datatype::INT32;

    for (size_t i = 0; i < kernel_size; i++) {
        auto dispatchData = SetDefault(newParams, /*is_second=*/i == 1);
        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, i);
        clKernelData& kernel = kd.kernels[i];

        cldnn_jit.RemoveConstant("ITER");
        cldnn_jit.AddConstant(MakeJitConstant("ITER", static_cast<int32_t>(i)));

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
                         params.is_shape_agnostic);

        // internal fixed-point accumulator buffer, every stage touches it
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});

        if (i == 1) {
            // local staging window for the update stage only
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 0});
            kernel.params.local_memory_args.push_back(kWindowSize * sizeof(int32_t));
        }
    }

    return {kd};
}

}  // namespace kernel_selector
