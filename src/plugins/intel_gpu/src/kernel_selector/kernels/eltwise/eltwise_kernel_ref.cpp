// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey EltwiseKernelRef::GetSupportedKey() const {
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
    k.EnableDifferentTypes();
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableEltwiseStride();
    k.EnableEltwiseBroadcast();
    return k;
}

bool EltwiseKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (!EltwiseKernelBase::Validate(p, o)) {
        return false;
    }

    return true;
}

KernelsData EltwiseKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetCommonKernelsData(params, options);
}

KernelsPriority EltwiseKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

JitConstants EltwiseKernelRef::GetJitConstants(const eltwise_params& params) const {
    auto jit = EltwiseKernelBase::GetJitConstants(params);

    if (!params.fused_ops.empty()) {
        kernel_selector::Datatype input_dt = GetAccumulatorType(params);

        std::vector<std::string> idx_order;
        if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 4) {
            if (!params.layoutBased && !params.int8_quantization && !params.broadcast && !CheckInputsOutputNoPitchSameDims(params)) {
                auto calc_dim = [&params](Tensor::DataChannelName channel) {
                    size_t idx = DataTensor::Channelndex(params.outputs[0].GetLayout(), channel);
                    // We increment the index, because fusions dims ordering starts from one
                    return "d" + std::to_string(idx + 1);
                };

                idx_order = {calc_dim(Tensor::DataChannelName::BATCH),
                             calc_dim(Tensor::DataChannelName::FEATURE),
                             calc_dim(Tensor::DataChannelName::Y),
                             calc_dim(Tensor::DataChannelName::X)};
            } else {
                idx_order = {"d4", "d3", "d2", "d1"};
            }
        } else if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 5) {
            idx_order = {"d5", "d4", "d3", "d2", "d1"};
        } else if (DataTensor::ChannelsCount(params.outputs[0].GetLayout()) == 6) {
            idx_order = {"d6", "d5", "d4", "d3", "d2", "d1"};
        }

        if (!params.layoutBased && !params.int8_quantization && !params.broadcast && CheckInputsOutputNoPitchSameDims(params)) {
            FusedOpsConfiguration conf = {"", {"d1"}, "res", input_dt, 1, LoadType::LT_UNALIGNED, BoundaryCheck::ENABLED, IndexType::LINEAR_OFFSET};
            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        } else {
            FusedOpsConfiguration conf =  {"", idx_order, "res", input_dt, 1};
            jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
        }
    }

    return jit;
}
}  // namespace kernel_selector
