// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "openvino/core/type/element_type.hpp"

namespace kernel_selector {
struct multiclass_nms_params : public base_params {
    multiclass_nms_params()
        : base_params(KernelType::MULTICLASS_NMS) {}

    SortResultType sort_result_type = SortResultType::CLASSID;
    bool sort_result_across_batch = false;
    float iou_threshold = 0.0f;
    float score_threshold = 0.0f;
    int nms_top_k = 0;
    int keep_top_k = 0;
    int background_class = 0;
    bool normalized = false;
    float nms_eta = 0.0f;
    bool has_roisnum = false;
};

class MulticlassNmsKernelRef : public KernelBaseOpenCL {
public:
    MulticlassNmsKernelRef() : KernelBaseOpenCL("multiclass_nms_ref") {}

    ~MulticlassNmsKernelRef() = default;

    using DispatchData = CommonDispatchData;

protected:
    bool Validate(const Params& p) const override;

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    void SetKernelArguments(const multiclass_nms_params& params, size_t idx, cldnn::arguments_desc& kernel) const;

private:
    JitConstants GetJitConstants(const multiclass_nms_params& params) const;
};
}  // namespace kernel_selector
