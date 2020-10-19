// Copyright (c) 2018-2020 Intel Corporation
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


#include "detection_output_kernel_ref.h"
#include "kernel_selector_utils.h"

#define PRIOR_BOX_SIZE 4  // Each prior-box consists of [xmin, ymin, xmax, ymax].

namespace kernel_selector {

ParamsKey DetectionOutputKernel::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

CommonDispatchData DetectionOutputKernel::SetDefault(const detection_output_params& params) const {
    CommonDispatchData dispatchData = DetectionOutputKernelBase::SetDefault(params);

    // Number of all work items is set to total number of bounding boxes -
    // one bounding box is procerssed by one work item
    size_t num_classes = (params.detectOutParams.share_location) ? 1 : params.detectOutParams.num_classes;

    // Size of input0 (input location), if shared loaction it is equal to size of one class,
    // else it has size of all items for all classes
    size_t bboxesNum = params.inputs[0].LogicalSize() / PRIOR_BOX_SIZE / num_classes;
    // Work group size is set to number of bounding boxes per image for sorting purpose
    // (access to one table with sorted values)
    size_t work_group_size = bboxesNum / params.inputs[0].Batch().v;

    if (work_group_size > 256) {
        work_group_size = work_group_size / ((work_group_size / 256) + 1) + 1;
    }

    bboxesNum = work_group_size * params.inputs[0].Batch().v;

    dispatchData.gws[0] = Align(bboxesNum, work_group_size);
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = 1;

    dispatchData.lws[0] = work_group_size;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsData DetectionOutputKernel::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::DETECTION_OUTPUT && options.GetType() == KernelType::DETECTION_OUTPUT);

    KernelData kd = KernelData::Default<detection_output_params>(params);
    const detection_output_params& detectOutParams = static_cast<const detection_output_params&>(params);
    DispatchData dispatchData = SetDefault(detectOutParams);

    auto cldnnJit = GetJitConstants(detectOutParams);
    auto entryPoint = GetEntryPoint(kernelName, detectOutParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entryPoint);
    kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
    kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});

    kd.estimatedTime = FORCE_PRIORITY_8;

    return {kd};
}
}  // namespace kernel_selector