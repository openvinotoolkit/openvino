/*
// Copyright (c) 2020 Intel Corporation
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

#include "space_to_depth_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
    ParamsKey SpaceToDepthKernelRef::GetSupportedKey() const {
        ParamsKey k;
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();
        return k;
    }

    CommonDispatchData SpaceToDepthKernelRef::SetDefault(const space_to_depth_params& params,
                                                         const optional_params&) const {
        CommonDispatchData runInfo;

        std::vector<size_t> global = {params.output.Batch().v,
                                      params.output.Feature().v,
                                      params.output.Y().v * params.output.X().v};

        auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

        runInfo.gws0 = global[0];
        runInfo.gws1 = global[1];
        runInfo.gws2 = global[2];

        runInfo.lws0 = local[0];
        runInfo.lws1 = local[1];
        runInfo.lws2 = local[2];

        return runInfo;
    }

    JitConstants SpaceToDepthKernelRef::GetJitConstants(const space_to_depth_params& params) const {
        JitConstants jit = MakeBaseParamsJitConstants(params);

        const size_t block_size = params.block_size;
        const size_t squared_block_size = params.block_size * params.block_size;
        const size_t blocks_first_mode = (size_t)params.depth_mode;

        jit.AddConstant(MakeJitConstant("BLOCK_SIZE", block_size));
        jit.AddConstant(MakeJitConstant("SQUARED_BLOCK_SIZE", squared_block_size));
        jit.AddConstant(MakeJitConstant("BLOCKS_FIRST_MODE", blocks_first_mode));

        return jit;
    }

    KernelsData SpaceToDepthKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
        KernelData kd = KernelData::Default<space_to_depth_params>(params);
        space_to_depth_params& newParams = *static_cast<space_to_depth_params*>(kd.params.get());

        assert(params.GetType() == KernelType::SPACE_TO_DEPTH);

        auto runInfo = SetDefault(newParams, options);
        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
        auto cldnn_jit = GetJitConstants(newParams);
        std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];

        FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point);

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return {kd};
    }
}  // namespace kernel_selector
