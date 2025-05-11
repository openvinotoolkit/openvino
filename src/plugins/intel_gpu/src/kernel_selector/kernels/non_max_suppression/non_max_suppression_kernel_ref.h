// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// non_max_suppression_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct non_max_suppression_params : public base_params {
    non_max_suppression_params() : base_params(KernelType::NON_MAX_SUPPRESSION),
    box_encoding(BoxEncodingType::BOX_ENCODING_CORNER), sort_result_descending(true),
    num_select_per_class_type(base_params::ArgType::Constant), num_select_per_class(0),
    iou_threshold_type(base_params::ArgType::Constant), iou_threshold(0.0f),
    score_threshold_type(base_params::ArgType::Constant), score_threshold(0.0f),
    soft_nms_sigma_type(base_params::ArgType::Constant), soft_nms_sigma(0.0f) {}

    BoxEncodingType box_encoding;
    bool sort_result_descending;
    base_params::ArgType num_select_per_class_type;
    int num_select_per_class;
    base_params::ArgType iou_threshold_type;
    float iou_threshold;
    base_params::ArgType score_threshold_type;
    float score_threshold;
    base_params::ArgType soft_nms_sigma_type;
    float soft_nms_sigma;
    bool reuse_internal_buffer = false;
    NMSRotationType rotation = NMSRotationType::NONE;

    uint32_t GetIndexNumSelectPerClass() const {
        uint32_t input_idx = 2;
        return input_idx;
    }

    uint32_t GetIndexIouThreshold() const {
        uint32_t input_idx = GetIndexNumSelectPerClass();
        if (num_select_per_class_type == base_params::ArgType::Input) input_idx++;
        return input_idx;
    }

    uint32_t GetIndexScoreThreshold() const {
        uint32_t input_idx = GetIndexIouThreshold();
        if (iou_threshold_type == base_params::ArgType::Input) input_idx++;
        return input_idx;
    }

    uint32_t GetIndexSoftNmsSigma() const {
        uint32_t input_idx = GetIndexScoreThreshold();
        if (score_threshold_type == base_params::ArgType::Input) input_idx++;
        return input_idx;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// NonMaxSuppressionKernelRef
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class NonMaxSuppressionKernelRef : public KernelBaseOpenCL {
public:
    NonMaxSuppressionKernelRef() : KernelBaseOpenCL("non_max_suppression_gpu_ref") {}

    using DispatchData = CommonDispatchData;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    Datatype GetAccumulatorType(const non_max_suppression_params& params) const;
    virtual JitConstants GetJitConstants(const non_max_suppression_params& params) const;
    bool Validate(const Params& p) const override;
    void SetKernelArguments(const non_max_suppression_params& params, clKernelData& kernel, size_t idx) const;
};

}  // namespace kernel_selector
