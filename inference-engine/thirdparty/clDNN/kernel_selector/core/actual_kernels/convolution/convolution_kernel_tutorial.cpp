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

#include "convolution_kernel_tutorial.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

        // Step 0: 
        //
        // take a look on convolution_kernel_tutorial.h 
    
    ParamsKey ConvolutionKernel_Tutorial::GetSupportedKey() const
    {
        // Step 1:
        // - Update the features supported by the kernel below

        ParamsKey k;
        
        // Supported data type
        k.EnableInputDataType(Datatype::F16);
        k.EnableInputDataType(Datatype::F32);
        k.EnableOutputDataType(Datatype::F16);
        k.EnableOutputDataType(Datatype::F32);
        k.EnableInputWeightsType(WeightsType::F16);
        k.EnableInputWeightsType(WeightsType::F32);

        // Supported layout
        k.EnableInputLayout(DataLayout::bfyx);
        k.EnableOutputLayout(DataLayout::bfyx);
        k.EnableInputLayout(DataLayout::yxfb);
        k.EnableOutputLayout(DataLayout::yxfb);

        // Supported tensor offset/pitch/padding
        k.EnableTensorOffset();
        k.EnableTensorPitches();
        k.EnableBatching();

        // Supported convolution extra data
        k.EnableDilation();
        k.EnableBiasPerFeature();
        k.EnableBiasPerOutput();
        k.EnableNonBiasTerm();

        // Supported convolution which get a split index and uses it as a view on the input/output
        k.EnableSplitSupport();

        // Supported convoltuion with depth separable optimization flag
        k.EnableDepthwiseSeparableOpt();

        return k;
    }

#ifdef BASIC_TUTORIAL

    KernelsData ConvolutionKernel_Tutorial::GetKernelsData(const Params& /*params*/, const optional_params& /*options*/) const
    {
        return{};

        // Step 2:
        // - Uncomment and update the following lines

        // assert(params.GetType() == KernelType::CONVOLUTION && options.GetType() == KernelType::CONVOLUTION);
        //
        // const uint32_t numOfkernels = 1;
        // KernelData kd = KernelData::Default<ConvolutionParams>(params, numOfkernels);
        // ConvolutionParams& newParams = *static_cast<ConvolutionParams*>(kd.params.get());
        // const ConvolutionOptionalParams& optParams = static_cast<const ConvolutionOptionalParams&>(options);
        // auto& kernel = kd.kernels[0];
        

        // Step 3:
        // - make sure that the input weights tensor fit to this kernel needs. 
        //   in case it's not and the flag "optParams.allowWeightsReorder" set to "true", please update
        //   the member "kd.weightsReorderParams" with the right OpenCL/CPU kernel which will be used to reorder the 
        //   weights in the loading time.
        //   you have three options:
        //   - provide a cpu code - inherit from "CPUKernel" and implement "Execute" function.
        //      (by default the input layout of CPU kernel is simple bfyx, and clDNN will reorder it for you before calling to Execute function)
        //   - provide a GPU code by filling clKernelData.
        //   - use existing layouts which clDNN support and use the auxiliary function "UpdateWeightsParams"


        // Step 4:
        // - make sure that the input tensor fits to this kernel's needs. 
        //   make sure that you have the proper padding area with a proper padding value, and a proper alignment.
        //   currently Convolution in clDNN doesn't allow the kernel to ask reordering


        // Step 5:
        // - fill "kernel.kernelString"
        //   - fill "kernel.kernelString->str"                  - the source of the kernel. 
        //     please use "db.get(kernelName)" in case you use "*.cl" file which located under "kernel_selector\core\cl_kernels\".
        //   - fill "kernel.kernelString->jit"                  - Dynamic jit of this params. 
        //   - fill "kernel.kernelString->options"              - options which pass to cl program build functions (like "-cl-no-subgroup-ifp")
        //   - fill "kernel.kernelString->entry_point"          - kernel entry point 
        //   - fill "kernel.kernelString->batch_compilation"    - A flag that allow clDNN kernel to compile this kernel as a part of a program
        //                                                        NOTE: this can only be used if you prevent symbol conflicts with other kernels (#undef is done automatically by clDNN)


        // Step 6:
        // - fill "kernel.WorkGroupSizes" - local/global work group sizes for OpenCL kernel


        // Step 7:
        // - fill "kernel.arguments" - which describe the argument of the kernel. 
        //   in this tutorial you can use:
        //     kernel.arguments.push_back({ ArgumentDescriptor::Types::INPUT, 0 }); // "0" mean index of the input in case of multiple inputs.
        //     kernel.arguments.push_back({ ArgumentDescriptor::Types::OUTPUT, 0 });
        //     kernel.arguments.push_back({ ArgumentDescriptor::Types::WEIGHTS, 0 });
        //     kernel.arguments.push_back({ ArgumentDescriptor::Types::BIAS, 0 });
        //
        //   in case that you have more than one kernel, you probably need an intermediate buffers.
        //   in order to support that you have to describe the buffer size in kd.internalBufferSizes and add a kernel argument like:
        //     kernel.arguments.push_back({ ArgumentDescriptor::Types::INTERNAL_BUFFER, <index to kd.internalBufferSize> });


        // Step 8:
        // - estimate the kernel's execution time. currently it's under development so please use FORCE_PRIORITY_<X> - lower is better.


        // return{ kd };
    }

#else

    ConvolutionKernel_Tutorial::Parent::DispatchData ConvolutionKernel_Tutorial::SetDefault(const convolution_params& params, int autoTuneIndex) const
    {
        DispatchData runInfo = Parent::SetDefault(params, autoTuneIndex);

        // Step 2:
        //
        // Init runInfo, and set kernel efficiency
        runInfo.effiency = TUTORIAL_PRIORITY;

        return runInfo;
    }

    bool ConvolutionKernel_Tutorial::Validate(const Params& p, const optional_params& o) const
    {
        if (!Parent::Validate(p, o))
        {
            return false;
        }

        // Step 3:
        // 
        // Validate this kernel support params and optional params. use:
        // const ConvolutionParams& params = static_cast<const ConvolutionParams&>(p);
        // const ConvolutionOptionalParams& options = static_cast<const ConvolutionOptionalParams&>(o);

        return true;
    }

    JitConstants ConvolutionKernel_Tutorial::GetJitConstants(const convolution_params& params, const DispatchData& kd) const
    {
        auto jit = Parent::GetJitConstants(params, kd);
        jit.AddConstant(MakeJitConstant("ADVANCED_TUTORIAL", ""));

        // Step 4:
        // 
        // Add you own jit constants. for example
        // jit.AddConstant(MakeJitConstant("<MY_CONST>", <my val>));
        // - "my val" can be most of KernelSelector/C++ common types

        return jit;
    }

    KernelsData ConvolutionKernel_Tutorial::GetKernelsData(const Params& params, const optional_params& options) const
    {
        return GetCommonKernelsData(params, options);
    }

#endif
}