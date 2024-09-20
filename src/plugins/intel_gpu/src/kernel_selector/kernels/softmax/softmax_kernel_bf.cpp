// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_kernel_bf.h"
#include "kernel_selector_utils.h"
#include <algorithm>

namespace kernel_selector {
static constexpr size_t subgroup_size = 16;
ParamsKey SoftmaxKernel_bf::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableSoftmaxDim(SoftmaxDim::X);  // in case that it can be flatten
    k.EnableSoftmaxDim(SoftmaxDim::Y);
    k.EnableSoftmaxDim(SoftmaxDim::FEATURE);
    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();
    return k;
}

DeviceFeaturesKey SoftmaxKernel_bf::get_required_device_features_key(const Params& params) const {
    DeviceFeaturesKey k;
    k.requires_subgroups();
    k.requires_subgroup_reduce();
    k.requires_reqd_subgroup_size();

    return k;
}

SoftmaxKernel_bf::Parent::DispatchData SoftmaxKernel_bf::SetDefault(const softmax_params& params) const {
    auto dispatchData = Parent::SetDefault(params);

    dispatchData.normIndex = 0;

    // We have two units of data per work item in current implementation.
    auto local_mem_per_wi = 2 * BytesPerElement(params.inputs[0].GetDType());
    // Combining device execution and local memory restrictions to compute maximum possible LWS.
    auto max_lws = std::min(params.engineInfo.maxWorkGroupSize, params.engineInfo.maxLocalMemSize / local_mem_per_wi);
    if (!params.has_dynamic_tensors()) {
        // start with 1 thread per data set
        dispatchData.gws[0] = 1;
        dispatchData.gws[1] = dispatchData.dataSetsCount;
        dispatchData.itemsNum = dispatchData.dataSetSize;

        dispatchData.lws[0] = 1;
        // Compute maximum possible LWS that does not exceed device capabilities and optimizes number of global memory
        // reads.
        while ((dispatchData.itemsNum > 32 || dispatchData.lws[0] < dispatchData.itemsNum) && (2 * dispatchData.lws[0] <= max_lws)) {
            dispatchData.lws[0] *= 2;
            dispatchData.itemsNum /= 2;
        }

        dispatchData.leftovers = dispatchData.dataSetSize % dispatchData.lws[0];
        // To use subgroup read/write, the starting address should be aligned to 128 bit
        size_t dataSetSizeInByte = dispatchData.dataSetSize * params.inputs[0].ElementSize();
        if ((dispatchData.dataSetsCount > 1) && ((dataSetSizeInByte - ((dataSetSizeInByte >> 4) << 4)))) {
            dispatchData.subgroupBlockSize = 1;
        } else {
            if (dispatchData.itemsNum >> 3)
                dispatchData.subgroupBlockSize = 8;
            else if (dispatchData.itemsNum >> 2)
                dispatchData.subgroupBlockSize = 4;
            else if (dispatchData.itemsNum >> 1)
                dispatchData.subgroupBlockSize = 2;
            else
                dispatchData.subgroupBlockSize = 1;
        }
        assert((dispatchData.itemsNum + 1) * dispatchData.lws[0] >= dispatchData.dataSetSize && "More than 'lws[0]' items per batch remains! Lws too small?");

        dispatchData.gws[0] = dispatchData.lws[0];

        assert(dispatchData.itemsNum > 0 && dispatchData.lws[0] && dispatchData.gws[0] > 0);
    } else {
        dispatchData.subgroupBlockSize = 1;
    }
    return dispatchData;
}

KernelsPriority SoftmaxKernel_bf::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_6;
}

void SoftmaxKernel_bf::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const softmax_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData SoftmaxKernel_bf::GetKernelsData(const Params& params) const {
    KernelsData kds = GetCommonKernelsData(params);
    if (!kds.empty()) {
        GetUpdateDispatchDataFunc(kds[0]);
    }

    return kds;
}

JitConstants SoftmaxKernel_bf::GetJitConstants(const softmax_params& params, DispatchData dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    if (params.has_dynamic_tensors()) {
        const auto& input = params.inputs[0];
        DimensionAccessHelperJit dims(input);
        auto softmax_dim_y_bfyx = (params.dim == SoftmaxDim::Y && input.GetLayout() == DataLayout::bfyx);
        auto softmax_dim_x_bfyx = (params.dim == SoftmaxDim::X && input.GetLayout() == DataLayout::bfyx);
        const std::string lws_0 = "get_local_size(0)";

        std::string data_set_count;
        std::string data_set_size;
        if (softmax_dim_y_bfyx) {
            data_set_count = toVectorMulString({dims.f(), dims.b()});
            data_set_size = dims.y();
        } else if (softmax_dim_x_bfyx) {
            data_set_count = toVectorMulString({dims.f(), dims.b(), dims.y()});
            data_set_size = dims.x();
        } else {
            data_set_count = dims.b();
            data_set_size = toVectorMulString({dims.x(), dims.y(), dims.z(), dims.f()});
        }

        // It can be expected that the maximum possible itemsNum will not exceed 32
        // Therefore, in dynamic shape, stack_size including additional buffer is set to 34(32 + 2(aligned offset + leftovers))
        constexpr size_t stack_size = 34; // The size of stack for my_chunk
        jit.AddConstants({
            MakeJitConstant("LWS", lws_0),
            MakeJitConstant("DATA_SETS_COUNT", data_set_count),
            MakeJitConstant("DATA_SET_SIZE", data_set_size),
            MakeJitConstant("STACK_SIZE", stack_size),
        });
    } else {
        jit.AddConstants({
            MakeJitConstant("ITEMS_NUM", dispatchData.itemsNum),
            MakeJitConstant("LWS", dispatchData.lws[0]),
            MakeJitConstant("DATA_SETS_COUNT", dispatchData.dataSetsCount),
            MakeJitConstant("DATA_SET_SIZE", dispatchData.dataSetSize),
            MakeJitConstant("STACK_SIZE", dispatchData.itemsNum + 2), // (aligned offset + leftovers)
        });
    }
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", subgroup_size));
    jit.AddConstant(MakeJitConstant("SUBGROUP_BLOCK_SIZE", dispatchData.subgroupBlockSize));
    auto activation_dt = GetActivationType(params);
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));

    if (!params.fused_ops.empty()) {
        FusedOpsConfiguration conf_main = {"_MAIN",
                                           {"data_set_offset", "in_data_set_idx + i * workers_per_data_set", "0", "0"},
                                           "dequantized",
                                           activation_dt};
        FusedOpsConfiguration conf_leftovers = {"_LEFTOVERS",
                                                {"data_set_offset", "workers_per_data_set * ITEMS_NUM + in_data_set_idx", "0", "0"},
                                                "dequantized",
                                                activation_dt};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_main, conf_leftovers}));
    }

    return jit;
}
}  // namespace kernel_selector
