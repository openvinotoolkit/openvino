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

#include "gather_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector
{
    static int32_t GetGatherChannelIndex(const gather_params& params)
    {
        Tensor::DataChannelName name = Tensor::DataChannelName::X;

        switch (params.axis)
        {
            case GatherAxis::X:
                return 3;
            case GatherAxis::Y:
                return 2;
            case GatherAxis::FEATURE:
                return 1;
            case GatherAxis::BATCH:
                return 0;
            default: break;
        }

        return DataTensor::Channelndex(params.output.GetLayout(), name);
    }

    ParamsKey GatherKernelRef::GetSupportedKey() const
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
        k.EnableLookUpTableIndicesFormat(Datatype::F32);
        return k;
    }

    static size_t getPartSize(const gather_params& params, int32_t axis)
    {
        size_t partSize = 1;
        for (size_t i = params.inputs[0].Dimentions() - axis; i > 0; --i)
            partSize *= params.inputs[0].GetDims()[i-1].v;
        return partSize;
    }

    static size_t getNumberOfParts(const gather_params& params, size_t partSize)
    {
        return params.inputs[0].LogicalSize() / partSize;
    }

    static size_t getSliceSize(const gather_params& params, int32_t axis)
    {
        size_t numberOfItemsInSlice = 1;
        for (size_t i = params.inputs[0].Dimentions() - axis - 1; i > 0; --i)
            numberOfItemsInSlice *= params.inputs[0].GetDims()[i-1].v;
        return numberOfItemsInSlice;
    }

    CommonDispatchData GatherKernelRef::SetDefault(const gather_params& params, const optional_params&) const
    {
        CommonDispatchData runInfo;

        const int32_t axis = GetGatherChannelIndex(params);

        const size_t numberOfParts = params.inputs[0].LogicalSize() / getPartSize(params, axis);

        size_t gws = numberOfParts * params.inputs[1].LogicalSize();

        const size_t vectorSize = 16;

        runInfo.gws0 = Align(gws, vectorSize);
        runInfo.gws1 = 1;
        runInfo.gws2 = 1;

        runInfo.lws0 = vectorSize;
        runInfo.lws1 = 1;
        runInfo.lws2 = 1;

        runInfo.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

        return runInfo;
    }

    JitConstants GatherKernelRef::GetJitConstants(const gather_params& params) const
    {
        JitConstants jit = MakeBaseParamsJitConstants(params);

        int32_t axis = GetGatherChannelIndex(params);
        size_t partSize = getPartSize(params, axis);
        size_t sliceSize = getSliceSize(params, axis);
        size_t numberOfParts = getNumberOfParts(params, partSize);
        size_t numberOfIndexes = params.inputs[1].LogicalSize();

        jit.AddConstant(MakeJitConstant("AXIS", axis));
        jit.AddConstant(MakeJitConstant("PART_SIZE", partSize));
        jit.AddConstant(MakeJitConstant("SLICE_SIZE", sliceSize));
        jit.AddConstant(MakeJitConstant("PARTS_NUMBER", numberOfParts));
        jit.AddConstant(MakeJitConstant("COMPUTATIONAL_OPERATIONS_NUMBER", numberOfParts * numberOfIndexes));

        return jit;
    }

    KernelsData GatherKernelRef::GetKernelsData(const Params& params, const optional_params& options) const
    {
        KernelData kd = KernelData::Default<gather_params>(params);
        gather_params& newParams = *static_cast<gather_params*>(kd.params.get());

        assert(params.GetType() == KernelType::GATHER);

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
