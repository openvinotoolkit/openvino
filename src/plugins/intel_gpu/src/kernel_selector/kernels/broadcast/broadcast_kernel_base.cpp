// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_kernel_base.h"
#include <vector>
#include "kernel_selector_utils.h"

namespace kernel_selector {

// Tune params to the specific platform.
#define VEC_SIZE 2
#define Y_BLOCKS 4

static bool is_same_planar_format(kernel_selector::Tensor::DataLayout in_layout, kernel_selector::Tensor::DataLayout out_layout) {
    return ((out_layout == DataLayout::bfyx && in_layout == DataLayout::bfyx)
            || (out_layout == DataLayout::bfzyx && in_layout == DataLayout::bfzyx)
            || (out_layout == DataLayout::bfwzyx && in_layout == DataLayout::bfwzyx));
}

JitConstants BroadcastKernelBase::GetJitConstants(const broadcast_params& params) const {
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();

    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({MakeJitConstant("BROADCAST_ORDER", params.input_order)});
    std::vector<uint16_t> default_order(params.input_order.size());
    std::iota(default_order.begin(), default_order.end(), 0);
    jit.AddConstants({MakeJitConstant("BROADCAST_ORDER_DEFAULT", default_order == params.input_order ? 1 : 0)});

    jit.AddConstants({MakeJitConstant("VEC_SIZE", VEC_SIZE)});
    jit.AddConstants({MakeJitConstant("Y_BLOCKS", Y_BLOCKS)});
    jit.AddConstants({MakeJitConstant("SAME_RANK_PLAIN_FORMAT", is_same_planar_format(in_layout, out_layout))});
    return jit;
}

BroadcastKernelBase::DispatchData BroadcastKernelBase::SetDefault(const broadcast_params& params) {
    const auto& output = params.outputs[0];
    const auto& input = params.inputs[0];

    DispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();

    auto is_broadcast_per_y_axis = [&]() {
        return is_same_planar_format(in_layout, out_layout)
            && (input.X().v == output.X().v) && (input.Y().v != output.Y().v)
            && (input.Batch().v == output.Batch().v) && (input.Feature().v == output.Feature().v)
            && (input.W().v == output.W().v) && (input.Z().v == output.Z().v);
    };

    // Use block calculation only when y is broadcast axis ans x dim is bigger than vec_size
    if (is_broadcast_per_y_axis()) {
        std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X },
                                                                        { Tensor::DataChannelName::Y },
                                                                        { Tensor::DataChannelName::Z, Tensor::DataChannelName::W,
                                                                        Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};
        dispatchData.gws = { ((output.X().v > VEC_SIZE)? (output.X().v / VEC_SIZE) : 1),
                             ((output.Y().v > Y_BLOCKS)? (output.Y().v / Y_BLOCKS) : 1),
                                output.Z().v * output.W().v * output.Feature().v * output.Batch().v };
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);
    } else {
        std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X },
                                                                        { Tensor::DataChannelName::Y, Tensor::DataChannelName::Z, Tensor::DataChannelName::W },
                                                                        { Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};

        dispatchData.gws = { output.X().v, output.Y().v * output.Z().v * output.W().v, output.Batch().v * output.Feature().v };
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);
    }

    return dispatchData;
}

static std::string GetInputBlockND(const broadcast_params& params) {
    const auto& input = params.inputs[0];

    std::stringstream s;
    auto input_dims = input.LogicalDims();
    std::reverse(input_dims.begin(), input_dims.end());

    if (input.is_dynamic()) {
        const int rank = static_cast<int>(input_dims.size());
        std::vector<std::string> block_nd_s(rank + 1);
        block_nd_s[rank] = "1";
        for (int idx = (rank - 1); idx >= 0; idx--) {
            int shape_info_idx = idx;
            if (idx >= 2) {
                shape_info_idx += (static_cast<int>(DataTensor::max_rank()) - rank);
            }
            block_nd_s[idx] = "(" + toCodeString(input.GetDims()[rank - idx - 1], shape_info_idx) + " * " + block_nd_s[idx + 1] + ")";
        }

        for (int i = 0; i < (rank + 1); i++) {
            s << block_nd_s[i];
            if (i < rank) {
                s << ",";
            }
        }
    } else {
        const int rank = static_cast<int>(input_dims.size());
        std::vector<size_t> block_nd(rank + 1);
        block_nd[rank] = 1;
        for (int idx = (rank - 1); idx >= 0; idx--) {
            block_nd[idx] = input_dims[idx] * block_nd[idx + 1];
        }

        for (int i = 0; i < (rank + 1); i++) {
            s << std::to_string(block_nd[i]);
            if (i < rank) {
                s << ",";
            }
        }
    }

    auto str_result = s.str();
    return str_result;
}

void BroadcastKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const broadcast_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData BroadcastKernelBase::GetCommonKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::BROADCAST);

    const auto& prim_params = static_cast<const broadcast_params&>(params);

    auto dispatchData = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<broadcast_params>(params);
    GetUpdateDispatchDataFunc(k_data);

    auto cldnn_jit = GetJitConstants(prim_params);
    cldnn_jit.AddConstant(MakeJitConstant("INPUT0_BLOCK_ND", GetInputBlockND(prim_params)));
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     EXE_MODE_DEFAULT,
                     false,
                     false,
                     1,
                     0,
                     1,
                     prim_params.is_shape_agnostic);

    return {k_data};
}
}  // namespace kernel_selector
