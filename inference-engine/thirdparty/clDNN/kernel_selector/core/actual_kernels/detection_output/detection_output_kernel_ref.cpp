/*
// Copyright (c) 2021 Intel Corporation
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

#include "detection_output_kernel_ref.h"
#include "kernel_selector_utils.h"

#define PRIOR_BOX_SIZE 4  // Each prior-box consists of [xmin, ymin, xmax, ymax].

namespace kernel_selector {

ParamsKey DetectionOutputKernelRef::GetSupportedKey() const {
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

JitConstants DetectionOutputKernelRef::GetJitConstants(const detection_output_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    const auto& detectOutParams = params.detectOutParams;

    jit.AddConstants({
        MakeJitConstant("NUM_IMAGES", detectOutParams.num_images),
        MakeJitConstant("NUM_CLASSES", detectOutParams.num_classes),
        MakeJitConstant("KEEP_TOP_K", detectOutParams.keep_top_k),
        MakeJitConstant("TOP_K", detectOutParams.top_k),
        MakeJitConstant("BACKGROUND_LABEL_ID", detectOutParams.background_label_id),
        MakeJitConstant("CODE_TYPE", detectOutParams.code_type),
        MakeJitConstant("CONF_SIZE_X", detectOutParams.conf_size_x),
        MakeJitConstant("CONF_SIZE_Y", detectOutParams.conf_size_y),
        MakeJitConstant("CONF_PADDING_X", detectOutParams.conf_padding_x),
        MakeJitConstant("CONF_PADDING_Y", detectOutParams.conf_padding_y),
        MakeJitConstant("SHARE_LOCATION", detectOutParams.share_location),
        MakeJitConstant("VARIANCE_ENCODED_IN_TARGET", detectOutParams.variance_encoded_in_target),
        MakeJitConstant("NMS_THRESHOLD", detectOutParams.nms_threshold),
        MakeJitConstant("ETA", detectOutParams.eta),
        MakeJitConstant("CONFIDENCE_THRESHOLD", detectOutParams.confidence_threshold),
        MakeJitConstant("IMAGE_WIDTH", detectOutParams.input_width),
        MakeJitConstant("IMAGE_HEIGH", detectOutParams.input_heigh),
        MakeJitConstant("ELEMENTS_PER_THREAD", detectOutParams.elements_per_thread),
        MakeJitConstant("PRIOR_COORD_OFFSET", detectOutParams.prior_coordinates_offset),
        MakeJitConstant("PRIOR_INFO_SIZE", detectOutParams.prior_info_size),
        MakeJitConstant("PRIOR_IS_NORMALIZED", detectOutParams.prior_is_normalized),
    });

    return jit;
}

DetectionOutputKernelRef::DispatchData SetDefault(const detection_output_params& params) {
    DetectionOutputKernelRef::DispatchData dispatchData;

    // // Number of all work items is set to total number of bounding boxes -
    // // one bounding box is procerssed by one work item
    // size_t num_classes = (params.detectOutParams.share_location) ? 1 : params.detectOutParams.num_classes;

    // // Size of input0 (input location), if shared loaction it is equal to size of one class,
    // // else it has size of all items for all classes
    // size_t bboxesNum = params.inputs[0].LogicalSize() / PRIOR_BOX_SIZE / num_classes;
    // // Work group size is set to number of bounding boxes per image for sorting purpose
    // // (access to one table with sorted values)
    // size_t work_group_size = bboxesNum / params.inputs[0].Batch().v;

    // if (work_group_size > 256) {
    //     work_group_size = work_group_size / ((work_group_size / 256) + 1) + 1;
    // }

    // bboxesNum = work_group_size * params.inputs[0].Batch().v;

    // dispatchData.gws[0] = Align(bboxesNum, work_group_size);
    // dispatchData.gws[1] = 1;
    // dispatchData.gws[2] = 1;

    // dispatchData.lws[0] = work_group_size;
    // dispatchData.lws[1] = 1;
    // dispatchData.lws[2] = 1;
    dispatchData.gws = { 1, 1, 1};
    dispatchData.lws = { 1, 1, 1};

    return dispatchData;
}

KernelsData DetectionOutputKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
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

    return {kd};
}

KernelsPriority DetectionOutputKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
