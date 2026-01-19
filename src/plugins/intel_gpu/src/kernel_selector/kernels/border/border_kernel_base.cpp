// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "border_kernel_base.h"
#include <vector>
#include "kernel_selector_utils.h"

namespace kernel_selector {

inline std::string GetInputTypeStr(uint32_t idx) {
    return "INPUT" + std::to_string(idx) + "_TYPE";
}

JitConstants BorderKernelBase::GetJitConstants(const border_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    uint32_t input_offset = 1;
    if (params.begin_type == base_params::ArgType::Input) {
        jit.AddConstant(MakeJitConstant("BEGIN_TYPE", GetInputTypeStr(input_offset)));
        input_offset += 1;
    } else {
        jit.AddConstant(MakeJitConstant("LT_SIZES", params.lt_sizes));
    }

    if (params.end_type == base_params::ArgType::Input) {
        jit.AddConstant(MakeJitConstant("END_TYPE", GetInputTypeStr(input_offset)));
        input_offset += 1;
    } else {
        jit.AddConstant(MakeJitConstant("RB_SIZES", params.rb_sizes));
    }

    if (params.pad_value_type == base_params::ArgType::Input) {
        jit.AddConstant(MakeJitConstant("BORDER_VALUE_TYPE", GetInputTypeStr(input_offset)));
        input_offset += 1;
    } else {
        jit.AddConstant(MakeJitConstant("BORDER_VALUE", params.border_value));
    }

    jit.AddConstants({MakeJitConstant(toString(params.b_type), "")});

    return jit;
}

BorderKernelBase::DispatchData BorderKernelBase::SetDefault(const border_params& params) const {
    const auto& output = params.outputs[0];

    DispatchData dispatchData;
    if (!params.has_dynamic_tensors()) {
        auto in_layout = params.inputs[0].GetLayout();
        auto out_layout = params.outputs[0].GetLayout();
        std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X, Tensor::DataChannelName::Z },
                                                                         { Tensor::DataChannelName::Y, Tensor::DataChannelName::W },
                                                                         { Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};

        dispatchData.gws = { output.X().v * output.Z().v, output.Y().v * output.W().v, output.Batch().v * output.Feature().v };
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);
    }

    using DispatchDataPair = std::pair<std::array<size_t, 3>, std::array<size_t, 3>>;
    static const std::array<DispatchDataPair, 4> lws_mapping_table = {{
            DispatchDataPair{{264, 264,  32}, {1, 1, 32}},
            DispatchDataPair{{264, 264,  64}, {1, 1, 64}},
            DispatchDataPair{{132, 132, 128}, {1, 1, 64}},
            DispatchDataPair{{  3, 336, 336}, {3, 8, 8}}
        }};
    std::array<size_t, 3> gws = {dispatchData.gws[0], dispatchData.gws[1], dispatchData.gws[2]};
    auto it = std::find_if(lws_mapping_table.begin(), lws_mapping_table.end(), [&gws](const DispatchDataPair& ddp) { return ddp.first == gws; });
    if (it != lws_mapping_table.end()) { dispatchData.lws = {it->second[0], it->second[1], it->second[2]}; }

    return dispatchData;
}

bool BorderKernelBase::SkipKernelExecution(const border_params& params) const {
    return params.outputs[0].LogicalSize() == 0;
}

void BorderKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const border_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = SkipKernelExecution(prim_params);
    };
}

KernelsData BorderKernelBase::GetCommonKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::BORDER);

    const auto& prim_params =
        static_cast<const border_params&>(params);

    auto dispatchData = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<border_params>(params);
    GetUpdateDispatchDataFunc(k_data);

    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    kernel.skip_execution = SkipKernelExecution(prim_params);

    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     EXE_MODE_DEFAULT,
                     false,
                     false,
                     (uint32_t)prim_params.inputs.size(),
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     prim_params.is_shape_agnostic);

    return {k_data};
}
}  // namespace kernel_selector
