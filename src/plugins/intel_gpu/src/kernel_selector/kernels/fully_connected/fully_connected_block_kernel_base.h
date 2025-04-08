// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "fully_connected_kernel_base.h"


namespace kernel_selector {
class FullyConnectedBlockKernelBase : public FullyConnectedKernelBase {
public:
    using FullyConnectedKernelBase::FullyConnectedKernelBase;
    virtual ~FullyConnectedBlockKernelBase() {}

protected:
    JitConstants GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const override;

    // how many batches will a single work item compute
    virtual size_t GetBatchesPerWorkItem(const fully_connected_params& params) const;

    size_t GetLocalGroupsSize(const fully_connected_params& params) const;

    // how many neurons for a single batch will a single work item produce
    static size_t GetNeuronsPerWorkItem(const fully_connected_params& params) {
        auto batchSize = params.outputs[0].Batch().v;
        auto out_elements_count_per_batch = params.outputs[0].LogicalSize() / batchSize;
        if (out_elements_count_per_batch % 16 == 0)
            return 2;
        else
            return 1;
    }
};
}  // namespace kernel_selector
