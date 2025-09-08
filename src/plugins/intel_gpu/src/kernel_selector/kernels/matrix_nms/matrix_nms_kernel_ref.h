// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {

struct matrix_nms_params : public base_params {
    matrix_nms_params() : base_params(KernelType::MATRIX_NMS) {}

    enum decay_function { GAUSSIAN, LINEAR };

    enum sort_result_type {
        CLASS_ID,  // sort selected boxes by class id (ascending) in each batch element
        SCORE,     // sort selected boxes by score (descending) in each batch element
        NONE       // do not guarantee the order in each batch element
    };

    // specifies order of output elements
    sort_result_type sort_type = sort_result_type::NONE;
    // specifies whenever it is necessary to sort selected boxes across batches or not
    bool sort_result_across_batch = false;
    // specifies minimum score to consider box for the processing
    float score_threshold = 0.0f;
    // specifies maximum number of boxes to be selected per class, -1 meaning to
    // keep all boxes
    int nms_top_k = -1;
    // specifies maximum number of boxes to be selected per batch element, -1
    // meaning to keep all boxes
    int keep_top_k = -1;
    // specifies the background class id, -1 meaning to keep all classes
    int background_class = -1;
    // specifies decay function used to decay scores
    decay_function decay = decay_function::LINEAR;
    // specifies gaussian_sigma parameter for gaussian decay_function
    float gaussian_sigma = 2.0f;
    // specifies threshold to filter out boxes with low confidence score after
    // decaying
    float post_threshold = 0.0f;
    // specifies whether boxes are normalized or not
    bool normalized = true;
};

class MatrixNmsKernelRef : public KernelBaseOpenCL {
public:
    MatrixNmsKernelRef() : KernelBaseOpenCL("matrix_nms_ref") {}

    using DispatchData = CommonDispatchData;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const matrix_nms_params& params) const;
    bool Validate(const Params& p) const override;
    void SetKernelArguments(const matrix_nms_params& params, clKernelData& kernel, size_t idx) const;
};

}  // namespace kernel_selector
