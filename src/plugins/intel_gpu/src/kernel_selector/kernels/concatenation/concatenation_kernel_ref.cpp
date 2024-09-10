// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "concatenation_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <string>

namespace kernel_selector {

ParamsKey ConcatenationKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableInputLayout(DataLayout::fb);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::fyxb);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::fb);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::fyxb);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableConcatAxis(ConcatAxis::X);
    k.EnableConcatAxis(ConcatAxis::Y);
    k.EnableConcatAxis(ConcatAxis::FEATURE);
    k.EnableConcatAxis(ConcatAxis::BATCH);
    k.EnableConcatKernelPerInput();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

JitConstants ConcatenationKernelRef::GetJitConstants(const concatenation_params& params) const {
    auto cldnnJit = ConcatenationKernelBase::GetJitConstants(params);
    auto input_format = params.inputs[0].GetLayout();

    if (params.inputs[0].Feature().v != 1) {
        cldnnJit.AddConstant(MakeJitConstant("CHECK_FEATURES", 1));
        int f_channel = DataTensor::Channelndex(params.outputs[0].GetLayout(), Tensor::DataChannelName::FEATURE);
        cldnnJit.AddConstant(MakeJitConstant("FEATURE_CHANNEL", f_channel));
    }

    // default values when input_format = output_format
    // d3 = batch, d2 = feature, d1 = y, d0 = x
    std::vector<std::string> dims_id = {"d3", "d2", "d1", "d0"};
    auto axis = ConcatenationKernelBase::GetConcatChannel(params);

    std::vector<Tensor::DataChannelName> axis_order = { Tensor::DataChannelName::BATCH,
                                                        Tensor::DataChannelName::FEATURE,
                                                        Tensor::DataChannelName::Y,
                                                        Tensor::DataChannelName::X };

    std::string input_dims_order = "";
    std::string output_dims_order = "";

    for (size_t i = 0; i < dims_id.size(); ++i) {
        std::string separator = i == dims_id.size() - 1 ? "" : ",";
        input_dims_order += dims_id[i] + separator;

        if (axis_order[i] == axis) {
            output_dims_order += "(" + dims_id[i] + " + output_offset_in_concat_axis)" + separator;
        } else {
            output_dims_order += dims_id[i] + separator;
        }
    }

    cldnnJit.AddConstant(MakeJitConstant("INPUT_DIMS_ORDER", input_dims_order));
    cldnnJit.AddConstant(MakeJitConstant("OUTPUT_DIMS_ORDER", output_dims_order));

    cldnnJit.AddConstant(MakeJitConstant("INPUT_DIM_0", DataTensor::Channelndex(input_format, Tensor::DataChannelName::X)));

    if (!params.fused_ops.empty()) {
        auto idx_order = dims_id;
        size_t axis_idx = std::distance(axis_order.begin(), std::find(axis_order.begin(), axis_order.end(), axis));
        idx_order[axis_idx] = "(" + idx_order[axis_idx] + " + output_offset_in_concat_axis)";

        auto conf = FusedOpsConfiguration("", idx_order, "result", params.inputs[0].GetDType());
        cldnnJit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }
    return cldnnJit;
}

KernelsData ConcatenationKernelRef::GetKernelsData(const Params& params) const {
    KernelsData kd = GetCommonKernelsData(params);

    if (!kd.empty()) {
        for (int i = 0; i < static_cast<int>(kd[0].kernels.size()); i++) {
            auto& kernel = kd[0].kernels[i];

            // to avoid cases when we execute with local work sizes 1x1x1
            if (kernel.params.workGroups.local[0] == 1 && kernel.params.workGroups.global[1] != 1) {
                kernel.params.workGroups.global[1] = Align(kernel.params.workGroups.global[1], 32);
                kernel.params.workGroups.local[1] = 32;
            }
        }
    }

    return kd;
}

KernelsPriority ConcatenationKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
