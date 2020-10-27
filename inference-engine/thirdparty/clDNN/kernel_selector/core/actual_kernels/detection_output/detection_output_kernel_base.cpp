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


#include "detection_output_kernel_base.h"

namespace kernel_selector {
JitConstants DetectionOutputKernelBase::GetJitConstants(const detection_output_params& params) const {
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

DetectionOutputKernelBase::DispatchData DetectionOutputKernelBase::SetDefault(const detection_output_params& /*params*/) const {
    DispatchData dispatchData;

    dispatchData.gws[0] = 0;
    dispatchData.gws[1] = 0;
    dispatchData.gws[2] = 0;

    dispatchData.lws[0] = 0;
    dispatchData.lws[1] = 0;
    dispatchData.lws[2] = 0;

    return dispatchData;
}
}  // namespace kernel_selector
