// Copyright (c) 2019 Intel Corporation
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

#include "gather_tree_kernel_base.h"
#include <vector>
#include "kernel_selector_utils.h"

namespace kernel_selector {
    JitConstants GatherTreeKernelBase::GetJitConstants(const gather_tree_params & params) const {
        JitConstants jit = MakeBaseParamsJitConstants(params);
        return jit;
    }

    GatherTreeKernelBase::DispatchData GatherTreeKernelBase::SetDefault(const gather_tree_params & params) const {
        std::vector<size_t> global{
                                    params.output.Y().v,  // beam
                                    params.output.Feature().v,  // batch
                                    1
                                  };
        const auto& local = GetOptimalLocalWorkGroupSizes(global);
        /*
            b -> time
            f -> batch
            y -> beam
        */
        DispatchData data;
        data.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;
        data.gws0 = global[0];
        data.gws1 = global[1];
        data.gws2 = global[2];
        data.lws0 = local[0];
        data.lws1 = local[1];
        data.lws2 = local[2];
        return data;
    }

    KernelsData GatherTreeKernelBase::GetCommonKernelsData(const Params& params,
                                                            const optional_params& options,
                                                            float estimated_time) const {
        assert(params.GetType() == KernelType::GATHER_TREE);
        const auto& gt_params = static_cast<const gather_tree_params&>(params);

        auto run_info = SetDefault(gt_params);
        auto kernel_data = KernelData::Default<gather_tree_params>(params);
        auto cldnn_jit = GetJitConstants(gt_params);
        auto entry_point = GetEntryPoint(kernelName, gt_params.layerID, options);
        auto jit = CreateJit(kernelName, cldnn_jit, entry_point);
        FillCLKernelData(kernel_data.kernels[0], run_info, params.engineInfo, kernelName, jit, entry_point, DEFAULT, false, false, 4);
        kernel_data.estimatedTime = estimated_time;
        return { kernel_data };
    }
}  // namespace kernel_selector
