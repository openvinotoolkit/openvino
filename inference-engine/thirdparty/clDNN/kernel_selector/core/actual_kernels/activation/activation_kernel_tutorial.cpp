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

#include "activation_kernel_tutorial.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

        // Step 0: 
        //
        // take a look on activaion_kernel_tutorial.h 

    ParamsKey ActivationKernel_Tutorial::GetSupportedKey() const
    {
        // Step 1:
        // - Update the features supported by the kernel below

        ParamsKey k;
        
        // Supported data type
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);

        // Supported layout
        k.EnableAllInputLayout();
        k.EnableAllOutputLayout();

        // Supported tensor offset/pitch/padding
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();

        // Supported activation extra data
        k.EnableActivationAdditionalParamsAsInput();

        return k;
    }

#ifdef BASIC_TUTORIAL

    KernelsData ActivationKernel_Tutorial::GetKernelsData(const Params& /*params*/, const optional_params& /*options*/) const
    {
        return{};

        // Step 2:
        // - Uncomment and update the following lines

        // assert(params.GetType() == KernelType::ACTIVATION && options.GetType() == KernelType::ACTIVATION);
        //
        // const uint32_t numOfkernels = 1;
        // KernelData kd = KernelData::Default<activation_params>(params, numOfkernels);
        // activation_params& newParams = *static_cast<activation_params*>(kd.params.get());
        // const activation_optional_params& optParams = static_cast<const activation_optional_params&>(options);
        // auto& kernel = kd.kernels[0];

        // Step 3:
        // - fill "kernel.kernelString"
        //   - fill "kernel.kernelString->str"                  - the source of the kernel. 
        //     please use "db.get(kernelName)" in case you use "*.cl" file which located under "kernel_selector\core\cl_kernels\".
        //   - fill "kernel.kernelString->jit"                  - Dynamic jit of this params. 
        //   - fill "kernel.kernelString->options"              - options which pass to cl program build functions (like "-cl-no-subgroup-ifp")
        //   - fill "kernel.kernelString->entry_point"          - kernel entry point 
        //   - fill "kernel.kernelString->batch_compilation"    - A flag that allow clDNN kernel to compile this kernel as a part of a program
        //                                                        NOTE: this can only be used if you prevent symbol conflicts with other kernels (#undef is done automatically by clDNN)

        // Step 4:
        // - fill "kernel.WorkGroupSizes" - local/global work group sizes for OpenCL kernel


        // Step 5:
        // - fill "kernel.arguments" - which describe the argument of the kernel. 
        //   in this tutorial you can use:
        //     kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 0 }); // "0" mean index of the input in case of multiple inputs.
        //     kernel.arguments.push_back({ ArgumentDescriptor::Types::OUTPUT, 0 });
        //
        //   in case that you have more than one kernel, you probably need an intermediate buffers.
        //   in order to support that you have to describe the buffer size in kd.internalBufferSizes and add a kernel argument like:
        //     kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, <index to kd.internalBufferSize> });


        // Step 6:
        // - estimate the kernel's execution time. currently it's under development so please use FORCE_PRIORITY_<X> - lower is better.


        // return{ kd };
    }

#else

    ActivationKernel_Tutorial::Parent::DispatchData ActivationKernel_Tutorial::SetDefault(const activation_params& params) const
    {
        auto runInfo = Parent::SetDefault(params);

        // Step 2:
        //
        // Init Dispatchdata, and set kernel effiecncy
        runInfo.effiency = TUTORIAL_PRIORITY;

        return runInfo;
    }

    bool ActivationKernel_Tutorial::Validate(const Params& p, const optional_params& o) const
    {
        if (!Parent::Validate(p, o))
        {
            return false;
        }

        // Step 3:
        // 
        // Validate this kernel support params and optional params. use:
        // const activation_params& params = static_cast<const activation_params&>(p);
        // const activation_optional_params& options = static_cast<const activation_optional_params&>(o);

        return true;
    }

    JitConstants ActivationKernel_Tutorial::GetJitConstants(const activation_params& params, DispatchData runInfo) const
    {
        auto jit = Parent::GetJitConstants(params, runInfo);
        jit.AddConstant(MakeJitConstant("ADVANCED_TUTORIAL", ""));

        // Step 4:
        // 
        // Add you own jit constants. for example
        // jit.AddConstant(MakeJitConstant("<MY_CONST>", <my val>));
        // - "my val" can be most of KernelSelector/C++ common types

        return jit;
    }

    KernelsData ActivationKernel_Tutorial::GetKernelsData(const Params& params, const optional_params& options) const
    {
        return GetCommonKernelsData(params, options);
    }

#endif
}