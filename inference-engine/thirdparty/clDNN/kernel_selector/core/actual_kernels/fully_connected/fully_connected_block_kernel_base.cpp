/*
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
*/

#include "fully_connected_block_kernel_base.h"

namespace kernel_selector 
{
    JitConstants FullyConnectedBlockKernelBase::GetJitConstants(const fully_connected_params& params, const FullyConnectedBlockKernelBase::DispatchData& data) const
    {
        auto cldnnJit = FullyConnectedKernelBase::GetJitConstants(params, data);

        const auto batches_per_work_item = GetBatchesPerWorkItem(params);

        cldnnJit.AddConstant(MakeJitConstant("NEURONS_PER_WORK_ITEM", GetNeuronsPerWorkItem(params))); // how many neurons for a single batch will a single work item produce
        cldnnJit.AddConstant(MakeJitConstant("BATCHES_PER_WORK_ITEM", batches_per_work_item));             // how many batches will a single work item compute
        cldnnJit.AddConstant(MakeJitConstant("OUTPUT_ELEMENTS_COUNT", params.output.LogicalSize() / params.output.Batch().v));

        return cldnnJit;
    }

}