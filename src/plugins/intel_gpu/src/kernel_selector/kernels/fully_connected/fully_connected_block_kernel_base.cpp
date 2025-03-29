// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_block_kernel_base.h"
#include <algorithm>

namespace kernel_selector {

    size_t FullyConnectedBlockKernelBase::GetBatchesPerWorkItem(const fully_connected_params& params) const {
        auto batchSize = params.outputs[0].Batch().v;
        return std::min(batchSize, static_cast<size_t>(32U));
    }

    size_t FullyConnectedBlockKernelBase::GetLocalGroupsSize(const fully_connected_params& params) const {
        auto batchSize = params.outputs[0].Batch().v;
        return std::max(static_cast<size_t>(1U), batchSize / GetBatchesPerWorkItem(params));
    }

JitConstants FullyConnectedBlockKernelBase::GetJitConstants(
    const fully_connected_params& params,
    const FullyConnectedBlockKernelBase::DispatchData& data) const {
    auto cldnnJit = FullyConnectedKernelBase::GetJitConstants(params, data);

    const auto batches_per_work_item = GetBatchesPerWorkItem(params);

    cldnnJit.AddConstant(MakeJitConstant(
        "NEURONS_PER_WORK_ITEM",
        GetNeuronsPerWorkItem(params)));  // how many neurons for a single batch will a single work item produce
    cldnnJit.AddConstant(MakeJitConstant("BATCHES_PER_WORK_ITEM",
                                         batches_per_work_item));  // how many batches will a single work item compute
    cldnnJit.AddConstant(
        MakeJitConstant("OUTPUT_ELEMENTS_COUNT", params.outputs[0].LogicalSize() / params.outputs[0].Batch().v));

    return cldnnJit;
}

}  // namespace kernel_selector
