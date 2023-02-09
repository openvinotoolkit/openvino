// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "openvino/core/type/element_type.hpp"

namespace kernel_selector {
struct multiclass_nms_params : public base_params {
    multiclass_nms_params()
        : base_params(KernelType::MULTICLASS_NMS) {}

    SortResultType sort_result_type;
    bool sort_result_across_batch;
    Datatype indices_output_type;
    float iou_threshold;
    float score_threshold;
    int nms_top_k;
    int keep_top_k;
    int background_class;
    bool normalized;
    float nms_eta;
    bool has_roisnum;
};

struct multiclass_nms_optional_params : public optional_params {
    multiclass_nms_optional_params()
        : optional_params(KernelType::MULTICLASS_NMS) {}
};

class MulticlassNmsKernelRef : public KernelBaseOpenCL {
public:
    MulticlassNmsKernelRef() : KernelBaseOpenCL("multiclass_nms_ref") {}

    ~MulticlassNmsKernelRef() = default;

    using DispatchData = CommonDispatchData;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    void SetKernelArguments(const multiclass_nms_params& params, size_t idx, cldnn::arguments_desc& kernel) const;

private:
    JitConstants GetJitConstants(const multiclass_nms_params& params) const;
};
}  // namespace kernel_selector
