// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_elements_update_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
static size_t GetScatterElementsUpdateChannelIndex(const scatter_elements_update_params& params) {
    Tensor::DataChannelName name = Tensor::DataChannelName::X;

    const size_t input_size = params.inputs[0].GetDims().size();
    switch (params.axis) {
        case ScatterUpdateAxis::X:
            return (size_t)(input_size - 1);
        case ScatterUpdateAxis::Y:
            return (size_t)(input_size - 2);
        case ScatterUpdateAxis::Z:
            return (size_t)(input_size - 3);
        case ScatterUpdateAxis::W:
            return 2;
        case ScatterUpdateAxis::FEATURE:
            return 1;
        case ScatterUpdateAxis::BATCH:
            return 0;
        default:
            break;
    }

    return DataTensor::Channelndex(params.outputs[0].GetLayout(), name);
}

ParamsKey ScatterElementsUpdateKernelRef::GetSupportedKey() const {
    ParamsKey k;
    const std::vector<Datatype> supportedTypes{
        Datatype::F16, Datatype::F32, Datatype::INT32, Datatype::INT8, Datatype::UINT8
    };
    for (const auto t : supportedTypes) {
        k.EnableInputDataType(t);
        k.EnableOutputDataType(t);
    }

    const std::vector<DataLayout> supportedLayots{
        DataLayout::bfyx,
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
        DataLayout::bfwzyx
    };
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

static inline std::vector<std::string> GetDefaultOrder(size_t size) {
    std::vector<std::string> default_order;
    if (size <= 4) {
        default_order = {"b", "f", "y", "x"};
    } else if (size == 5) {
        default_order = {"b", "f", "z", "y", "x"};
    } else if (size == 6) {
        default_order = {"b", "f", "w", "z", "y", "x"};
    }

    return default_order;
}

CommonDispatchData ScatterElementsUpdateKernelRef::SetDefault(const scatter_elements_update_params& params, bool is_second) const {
    CommonDispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws;

    const auto& output = params.outputs[0];
    const auto& indices = params.inputs[1];
    const auto& scope = is_second ? indices : output;

    const auto rank = params.inputs[0].GetDims().size();
    if (!scope.is_dynamic()) {
        if (is_second && params.mode != ScatterUpdateReduction::NONE) {
            switch (rank) {
                case 4:
                    dispatchData.gws = {indices.X().v * indices.Y().v, indices.Feature().v, indices.Batch().v};
                    dispatchData.lws = {1, 1, indices.Batch().v};
                    break;
                case 5:
                    dispatchData.gws = {indices.X().v * indices.Y().v, indices.Z().v * indices.Feature().v, indices.Batch().v};
                    dispatchData.lws = {1, 1, indices.Batch().v};
                    break;
                case 6:
                    dispatchData.gws = {indices.X().v * indices.Y().v, indices.Z().v * indices.W().v, indices.Feature().v * indices.Batch().v};
                    dispatchData.lws = {1, 1, indices.Batch().v};
                    break;
                default:
                    throw std::invalid_argument("Unsupported data layout for scatter elements update primitive");
                    break;
              }
        } else {
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

                case 6:
                    dispatchData.gws = {scope.X().v * scope.Y().v, scope.Z().v * scope.W().v, scope.Feature().v * scope.Batch().v};
                    dims_by_gws = {{Tensor::DataChannelName::X, Tensor::DataChannelName::Y},
                                  {Tensor::DataChannelName::Z, Tensor::DataChannelName::W},
                                  {Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH}};
                    break;
                default:
                    throw std::invalid_argument("Unsupported data layout for scatter elements update primitive");
                    break;
              }
              dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);
        }
    }

    return dispatchData;
}

JitConstants ScatterElementsUpdateKernelRef::GetJitConstants(const scatter_elements_update_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstant(MakeJitConstant("AXIS_VALUE", GetScatterElementsUpdateChannelIndex(params)));

    const auto& output = params.outputs[0];
    if (output.PhysicalSizeInBytes() * 4 > params.engineInfo.maxLocalMemSize) {
        jit.AddConstant(MakeJitConstant("NO_LOCAL_MEMORY", 1));
    }

    if (params.mode != ScatterUpdateReduction::NONE) {
        jit.AddConstant(MakeJitConstant("REDUCE_MODE", static_cast<int>(params.mode)));
        jit.AddConstant(MakeJitConstant("USE_INIT_VAL", params.use_init_val));
    }

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf1 = { "_FIRST_KERNEL", GetDefaultOrder(params.outputs[0].GetDims().size()), "val", params.inputs[0].GetDType() };
        FusedOpsConfiguration conf2 = { "_SECOND_KERNEL", GetDefaultOrder(params.outputs[0].GetDims().size()), "val", params.inputs[0].GetDType() };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf1, conf2}));
    }

    return jit;
}

bool ScatterElementsUpdateKernelRef::Validate(const Params& p) const {
    if (p.GetType() != KernelType:: SCATTER_ELEMENTS_UPDATE) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const scatter_elements_update_params& params = static_cast<const scatter_elements_update_params&>(p);

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    return true;
}

bool ScatterElementsUpdateKernelRef::SkipKernelExecution(const scatter_elements_update_params& params, size_t kernel_id) const {
    if (kernel_id == 0) {
        if (params.outputs[0].LogicalSize() != 0 && params.outputs[0] != params.inputs[0]) {
            return false;
        }
    }
    return KernelData::SkipKernelExecution(params);
}

void ScatterElementsUpdateKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const scatter_elements_update_params&>(params);

        if (prim_params.mode == ScatterUpdateReduction::NONE) {
            OPENVINO_ASSERT(kd.kernels.size() == 2, "[GPU] Invalid kernels size for update dispatch data func");
        } else {
            OPENVINO_ASSERT(kd.kernels.size() == 3, "[GPU] Invalid kernels size for update dispatch data func");
        }

        const auto& output = prim_params.outputs[0];
        const bool use_local_memory = (output.PhysicalSizeInBytes() * 4 > params.engineInfo.maxLocalMemSize) ? false : true;

        if (prim_params.mode != ScatterUpdateReduction::NONE) {
            kd.internalBuffers.clear();
            kd.internalBuffers.push_back(output.PhysicalSizeInBytes() * 2);

            if (!use_local_memory) {
                kd.internalBuffers.push_back(output.PhysicalSizeInBytes() * 2);
            }
            if (prim_params.mode == ScatterUpdateReduction::MEAN) {
                kd.internalBuffers.push_back(output.PhysicalSizeInBytes() * 2);
            }
            kd.internalBufferDataType = Datatype::INT32;
        }

        for (size_t i = 0; i < kd.kernels.size(); ++i) {
            auto dispatchData = SetDefault(prim_params, i == 1);
            kd.kernels[i].params.workGroups.global = dispatchData.gws;
            kd.kernels[i].params.workGroups.local = dispatchData.lws;
            kd.kernels[i].skip_execution = SkipKernelExecution(prim_params, i);

            if (i >= 1 && prim_params.mode != ScatterUpdateReduction::NONE && use_local_memory) {
                const auto& output = prim_params.outputs[0];
                const auto buffer_size = output.PhysicalSizeInBytes() * 2;
                kd.kernels[i].params.local_memory_args.clear();
                kd.kernels[i].params.local_memory_args.push_back(buffer_size);
            }
        }
    };
}

KernelsData ScatterElementsUpdateKernelRef::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const auto& prim_params = static_cast<const scatter_elements_update_params&>(params);
    int kernel_size = 2;

    if (prim_params.mode != ScatterUpdateReduction::NONE) {
        kernel_size += 1;
    }

    KernelData kd = KernelData::Default<scatter_elements_update_params>(params, kernel_size);
    scatter_elements_update_params& newParams = *static_cast<scatter_elements_update_params*>(kd.params.get());
    auto cldnn_jit = GetJitConstants(newParams);

    GetUpdateDispatchDataFunc(kd);

    const auto& output = newParams.outputs[0];
    const bool use_local_memory = (output.PhysicalSizeInBytes() * 4 > params.engineInfo.maxLocalMemSize) ? false : true;
    kd.internalBuffers.clear();
    kd.internalBuffers.push_back(output.PhysicalSizeInBytes() * 2); // fixed point output
    if (!use_local_memory) {
        kd.internalBuffers.push_back(output.PhysicalSizeInBytes() * 2); // reduction value output
        kd.internalBuffers.push_back(output.PhysicalSizeInBytes() * 2); // reduction_thread_count output
    }
    kd.internalBufferDataType = Datatype::INT32;
    if (newParams.mode == ScatterUpdateReduction::MEAN) {
        kd.internalBuffers.push_back(output.PhysicalSizeInBytes() * 2); // calculate mean
    }

    for (int i = 0; i < kernel_size; i++) {
        auto dispatchData = SetDefault(newParams, (i == 1));
        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, i);
        clKernelData& kernel = kd.kernels[i];

        cldnn_jit.RemoveConstant("ITER");
        cldnn_jit.AddConstant(MakeJitConstant("ITER", i));

        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

        if (i >= 1 && newParams.mode != ScatterUpdateReduction::NONE && use_local_memory) {
            const auto buffer_size = output.PhysicalSizeInBytes() * 2;
            kd.kernels[i].params.local_memory_args.clear();
            kd.kernels[i].params.local_memory_args.push_back(buffer_size);
        }

        FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, "", false, false, 3, GetFusedPrimitiveInputsCount(params), 1,
            params.is_shape_agnostic);

        uint32_t buf_idx = 0;

        if (newParams.mode != ScatterUpdateReduction::NONE) {
            // store output in fixed point
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, buf_idx++});
        }


        if (i >= 1 && newParams.mode != ScatterUpdateReduction::NONE) {
            if (use_local_memory) {
                // data reduction
                kernel.params.arguments.push_back({ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 0});
                kernel.params.arguments.push_back({ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 0});
            } else {
                // identify thread for perform write
                kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, buf_idx++});
            }
            if (newParams.mode == ScatterUpdateReduction::MEAN) {
                kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, buf_idx++});
            }
        }
    }

    return {kd};
}
}  // namespace kernel_selector
