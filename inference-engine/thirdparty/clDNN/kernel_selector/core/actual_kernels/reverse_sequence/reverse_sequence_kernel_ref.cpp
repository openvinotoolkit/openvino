/*
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
*/

#include "reverse_sequence_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector
{
    ParamsKey ReverseSequenceKernelRef::GetSupportedKey() const
    {
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
        k.EnableDifferentTypes();
        return k;
    }

    CommonDispatchData ReverseSequenceKernelRef::SetDefault(const reverse_sequence_params& params, const optional_params&) const
    {
        CommonDispatchData runInfo;

        std::vector<size_t> global = { params.output.Batch().v, params.output.Feature().v, params.output.Y().v * params.output.X().v };

        auto local = GetOptimalLocalWorkGroupSizes(global);

        runInfo.gws0 = global[0];
        runInfo.gws1 = global[1];
        runInfo.gws2 = global[2];

        runInfo.lws0 = local[0];
        runInfo.lws1 = local[1];
        runInfo.lws2 = local[2];

        return runInfo;
    }

    JitConstants ReverseSequenceKernelRef::GetJitConstants(const reverse_sequence_params& params) const
    {
        JitConstants jit = MakeBaseParamsJitConstants(params);

        jit.AddConstant(MakeJitConstant("SEQ_AXIS", params.seq_axis));
        jit.AddConstant(MakeJitConstant("BATCH_AXIS", params.batch_axis));

        return jit;
    }

    KernelsData ReverseSequenceKernelRef::GetKernelsData(const Params& params, const optional_params& options) const
    {
        KernelData kd = KernelData::Default<reverse_sequence_params>(params);
        reverse_sequence_params& newParams = *static_cast<reverse_sequence_params*>(kd.params.get());

        assert(params.GetType() == KernelType::REVERSE_SEQUENCE);

        auto runInfo = SetDefault(newParams, options);
        auto entry_point = GetEntryPoint(kernelName, newParams.layerID, options);
        auto cldnn_jit = GetJitConstants(newParams);
        std::string jit = CreateJit(kernelName, cldnn_jit, entry_point);

        auto& kernel = kd.kernels[0];

        FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entry_point, "", false, false, 2);

        kd.estimatedTime = DONT_USE_IF_HAVE_SOMETHING_ELSE;

        return{ kd };
    }
}
