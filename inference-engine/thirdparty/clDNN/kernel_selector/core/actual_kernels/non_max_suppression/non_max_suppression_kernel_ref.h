// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {

enum class NmsArgType {
    None,
    Input,
    Constant
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// non_max_suppression_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct non_max_suppression_params : public base_params {
    non_max_suppression_params() : base_params(KernelType::NON_MAX_SUPPRESSION),
    box_encoding(BoxEncodingType::BOX_ENCODING_CORNER), sort_result_descending(true),
    num_select_per_class_type(NmsArgType::None), num_select_per_class(0),
    iou_threshold_type(NmsArgType::None), iou_threshold(0.0f),
    score_threshold_type(NmsArgType::None), score_threshold(0.0f),
    soft_nms_sigma_type(NmsArgType::None), soft_nms_sigma(0.0f),
    has_second_output(false), has_third_output(false) {}

    BoxEncodingType box_encoding;
    bool sort_result_descending;
    NmsArgType num_select_per_class_type;
    int num_select_per_class;
    NmsArgType iou_threshold_type;
    float iou_threshold;
    NmsArgType score_threshold_type;
    float score_threshold;
    NmsArgType soft_nms_sigma_type;
    float soft_nms_sigma;
    bool has_second_output;
    bool has_third_output;

    virtual ParamsKey GetParamsKey() const {
        auto k = base_params::GetParamsKey();
        return k;
    }

    uint32_t GetIndexNumSelectPerClass() const {
        uint32_t input_idx = 2;
        return input_idx;
    }

    uint32_t GetIndexIouThreshold() const {
        uint32_t input_idx = 2;
        if (num_select_per_class_type == NmsArgType::Input) input_idx++;
        return input_idx;
    }

    uint32_t GetIndexScoreThreshold() const {
        uint32_t input_idx = 2;
        if (num_select_per_class_type == NmsArgType::Input) input_idx++;
        if (iou_threshold_type == NmsArgType::Input) input_idx++;
        return input_idx;
    }

    uint32_t GetIndexSoftNmsSigma() const {
        uint32_t input_idx = 2;
        if (num_select_per_class_type == NmsArgType::Input) input_idx++;
        if (iou_threshold_type == NmsArgType::Input) input_idx++;
        if (score_threshold_type == NmsArgType::Input) input_idx++;
        return input_idx;
    }

    uint32_t GetIndexSecondOutput() const {
        uint32_t input_idx = 2;
        if (num_select_per_class_type == NmsArgType::Input) input_idx++;
        if (iou_threshold_type == NmsArgType::Input) input_idx++;
        if (score_threshold_type == NmsArgType::Input) input_idx++;
        if (soft_nms_sigma_type == NmsArgType::Input) input_idx++;
        return input_idx;
    }

    uint32_t GetIndexThirdOutput() const {
        uint32_t input_idx = 2;
        if (num_select_per_class_type == NmsArgType::Input) input_idx++;
        if (iou_threshold_type == NmsArgType::Input) input_idx++;
        if (score_threshold_type == NmsArgType::Input) input_idx++;
        if (soft_nms_sigma_type == NmsArgType::Input) input_idx++;
        if (has_second_output) input_idx++;
        return input_idx;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// non_max_suppression_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct non_max_suppression_optional_params : optional_params {
    non_max_suppression_optional_params() : optional_params(KernelType::NON_MAX_SUPPRESSION) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// NonMaxSuppressionKernelRef
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class NonMaxSuppressionKernelRef : public KernelBaseOpenCL {
public:
    NonMaxSuppressionKernelRef() : KernelBaseOpenCL("non_max_suppression_gpu_ref") {}
    virtual ~NonMaxSuppressionKernelRef() {}

    using DispatchData = CommonDispatchData;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    Datatype GetAccumulatorType(const non_max_suppression_params& params) const;
    virtual JitConstants GetJitConstants(const non_max_suppression_params& params) const;
    bool Validate(const Params& p, const optional_params& o) const override;
    void SetKernelArguments(const non_max_suppression_params& params, clKernelData& kernel, size_t idx) const;
};

}  // namespace kernel_selector
