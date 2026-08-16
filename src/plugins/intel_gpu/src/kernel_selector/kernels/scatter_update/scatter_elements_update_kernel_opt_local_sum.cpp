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

// Mirrors scatter_elements_update_kernel_ref.cpp's file-local
// GetScatterElementsUpdateChannelIndex -- duplicated rather than shared since that
// function is file-static there and this kernel is deliberately independent of `_ref`
// (see the header comment for why).
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

KernelsPriority ScatterElementsUpdateKernelOptLocalSum::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}

ParamsKey ScatterElementsUpdateKernelOptLocalSum::GetSupportedKey() const {
    // Deliberately as broad as `_ref`'s own key (same supported dtypes/layouts) --
    // this is only a cheap pre-filter checked before GetKernelsData()/Validate() ever
    // run (see kernel_selector_base::GetAllImplementations); the real narrowing to
    // this kernel's actual scope (SUM mode, dense scatter, static shapes, etc.) lives
    // in Validate(), not here. Missing EnableDynamicShapesSupport() here specifically
    // caused this kernel to be silently filtered out before Validate() ever ran, for
    // every case tried -- confirmed via temporary debug logging in Validate() showing
    // zero calls; fixed by matching `_ref`'s key exactly.
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

bool ScatterElementsUpdateKernelOptLocalSum::Validate(const Params& p) const {
    if (p.GetType() != KernelType::SCATTER_ELEMENTS_UPDATE) {
        return false;
    }
    const auto& params = static_cast<const scatter_elements_update_params&>(p);

    // Scoped narrowly to the case this was designed and measured against -- never
    // replaces `_ref`'s correctness for anything outside this, only opts in for it
    // (this kernel is attached ahead of `_ref` in the selector; returning false here
    // falls through to `_ref` unchanged).
    if (params.mode != ScatterUpdateReduction::SUM) {
        return false;
    }
    if (!params.use_init_val) {
        return false;
    }
    if (params.is_shape_agnostic) {
        return false;  // static shapes only for this first version
    }
    if (!is_global_memory_case(params)) {
        return false;  // `_ref`'s own whole-output-fits-in-local-memory path already wins here
    }
    if (!params.fused_ops.empty()) {
        return false;  // keep the first version simple; `_ref` still handles fused cases
    }
    const size_t rank = params.inputs[0].GetDims().size();
    if (rank != 4 && rank != 5) {
        return false;
    }
    // Dense-ish scatter: the updates tensor is at least as large as the output --
    // covers both a 1:1 dense scatter and our real fused workload specifically (4
    // bilinear-corner update sets concatenated onto one output, so updates is 4x the
    // output there). Not a correctness requirement of the kernel itself (the
    // local/global-fallback split is safe for any index pattern/ratio), just a scope
    // guard against genuinely sparse scatters (a handful of indices into a huge
    // output) where zeroing/flushing a whole window per workgroup would be wasted
    // work for no real locality gain.
    if (params.inputs[2].LogicalSize() < params.outputs[0].LogicalSize()) {
        return false;
    }
    return true;
}

JitConstants ScatterElementsUpdateKernelOptLocalSum::GetJitConstants(
    const scatter_elements_update_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    jit.AddConstant(MakeJitConstant("AXIS_VALUE", GetChannelIndex(params)));
    jit.AddConstant(MakeJitConstant("WINDOW_SIZE", kWindowSize));
    // Element budget of the internal fixed-point accumulator buffer (see GetKernelsData's
    // matching `output.PhysicalSizeInBytes() * 2` byte-size allocation) -- the update
    // stage's write-back loop bound-checks against this so a window that straddles the
    // buffer's end can never write out of bounds, regardless of dtype-size rounding in
    // that byte formula.
    const size_t total_elements = (params.outputs[0].PhysicalSizeInBytes() * 2) / sizeof(int32_t);
    jit.AddConstant(MakeJitConstant("OPT_LOCAL_ACC_TOTAL_ELEMENTS", total_elements));
    return jit;
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

    // Two genuinely different dispatch layouts, matching `_ref`'s own SetDefault
    // exactly (mode is always SUM here, per Validate(), so `is_second` alone decides
    // which one applies). The update stage (is_second=true) merges X*Y into gws[0] as
    // ONE dispatch dimension -- the ITER==1 .cl body's `x = dim0 % INPUT2_SIZE_X; y =
    // dim0 / INPUT2_SIZE_X;` decoding only makes sense against THIS merged layout.
    // Using the non-merged (init/finalize) layout for the update stage too was a real
    // bug in an earlier version of this kernel: it silently scrambled x/y/f/b (e.g. a
    // dispatch shaped (X=1, Y=N) put every real thread index in what the update-stage
    // decode treated as `f`, not `y`), confirmed via direct dispatch-dimension dumps
    // showing all contributions landing on the same output element.
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

void ScatterElementsUpdateKernelOptLocalSum::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const scatter_elements_update_params&>(params);
        OPENVINO_ASSERT(kd.kernels.size() == 3, "[GPU] Invalid kernels size for scatter_elements_update_opt_local_sum");

        for (size_t i = 0; i < kd.kernels.size(); ++i) {
            // is_second==true selects the "update" (ITER==1) dispatch shape (sized to
            // the indices/updates tensor, which for us equals the output size anyway).
            auto dispatchData = this->SetDefault(prim_params, /*is_second=*/i == 1);
            kd.kernels[i].params.workGroups.global = dispatchData.gws;
            kd.kernels[i].params.workGroups.local = dispatchData.lws;
            kd.kernels[i].skip_execution = KernelData::SkipKernelExecution(prim_params, i);

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
    kd.internalBuffers.push_back(output.PhysicalSizeInBytes() * 2);  // fixed-point accumulator
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
