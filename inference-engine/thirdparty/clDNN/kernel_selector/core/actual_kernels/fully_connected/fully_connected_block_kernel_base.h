// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "fully_connected_kernel_base.h"


namespace kernel_selector {
class FullyConnectedBlockKernelBase : public FullyConnectedKernelBase {
public:
    using FullyConnectedKernelBase::FullyConnectedKernelBase;
    virtual ~FullyConnectedBlockKernelBase() {}

protected:
    JitConstants GetJitConstants(const fully_connected_params& params, const DispatchData& kd) const override;

    // how many batches will a single work item compute
    virtual size_t GetBatchesPerWorkItem(const fully_connected_params& params) const;

    size_t GetLocalGroupsSize(const fully_connected_params& params) const;

    // how many neurons for a single batch will a single work item produce
    static size_t GetNeuronsPerWorkItem(const fully_connected_params& params) {
        auto batchSize = params.output.Batch().v;
        auto out_elements_count_per_batch = params.output.LogicalSize() / batchSize;
        if (out_elements_count_per_batch % 16 == 0)
            return 2;
        else
            return 1;
    }
};
}  // namespace kernel_selector
