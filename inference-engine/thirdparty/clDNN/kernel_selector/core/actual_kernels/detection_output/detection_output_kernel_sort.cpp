/*
// Copyright (c) 2018 Intel Corporation
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

#include "detection_output_kernel_sort.h"
#include "kernel_selector_utils.h"

#define DETECTION_OUTPUT_ROW_SIZE 7 // Each detection consists of [image_id, label, confidence, xmin, ymin, xmax, ymax].

namespace kernel_selector
{

    ParamsKey DetectionOutputKernel_sort::GetSupportedKey() const
    {
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

    CommonDispatchData DetectionOutputKernel_sort::SetDefault(const detection_output_params& params) const
    {
        CommonDispatchData runInfo = DetectionOutputKernelBase::SetDefault(params);

        unsigned class_num = params.detectOutParams.num_classes;
        if (params.detectOutParams.share_location && params.detectOutParams.background_label_id == 0)
        {
            class_num -= 1;
        }
        const size_t bboxesNum = class_num * params.detectOutParams.num_images;
        // Work group size is set to number of bounding boxes per image
        size_t work_group_size = class_num;

        if (work_group_size > 256)
        {
            work_group_size = (work_group_size + work_group_size % 2) / (work_group_size / 256 + 1);
        }

        runInfo.gws0 = Align(bboxesNum, work_group_size);
        runInfo.gws1 = 1;
        runInfo.gws2 = 1;

        runInfo.lws0 = work_group_size;
        runInfo.lws1 = 1;
        runInfo.lws2 = 1;

        return runInfo;
    }

    KernelsData DetectionOutputKernel_sort::GetKernelsData(const Params& params, const optional_params& options) const
    {
        assert(params.GetType() == KernelType::DETECTION_OUTPUT &&
               options.GetType() == KernelType::DETECTION_OUTPUT);

        KernelData kd = KernelData::Default<detection_output_params>(params);
        const detection_output_params& detectOutParams = static_cast<const detection_output_params&>(params);
        DispatchData runInfo = SetDefault(detectOutParams);

        auto cldnnJit = GetJitConstants(detectOutParams);
        auto entryPoint = GetEntryPoint(kernelName, detectOutParams.layerID, options);
        auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel, runInfo, params.engineInfo, kernelName, jit, entryPoint);

        kd.estimatedTime = FORCE_PRIORITY_8;

        return{ kd };
    }
}