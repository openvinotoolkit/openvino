// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
struct experimental_detectron_generate_proposals_single_image_params : public base_params {
    experimental_detectron_generate_proposals_single_image_params()
    : base_params(KernelType::EXPERIMENTAL_DETECTRON_GENERATE_PROPOSALS_SINGLE_IMAGE) {}

    float min_size{0.0F};
    float nms_threshold{0.0F};
    size_t pre_nms_count{0};
    size_t post_nms_count{0};
};

class ExperimentalDetectronGenerateProposalsSingleImageRef : public KernelBaseOpenCL {
public:
    ExperimentalDetectronGenerateProposalsSingleImageRef()
    : KernelBaseOpenCL("experimental_detectron_generate_proposals_single_image_ref") {}

    ~ExperimentalDetectronGenerateProposalsSingleImageRef() override = default;

    using DispatchData = CommonDispatchData;

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
protected:
    bool Validate(const Params& p) const override;
    static void SetKernelArguments(const experimental_detectron_generate_proposals_single_image_params& params, size_t idx, cldnn::arguments_desc& arguments);
};
}  // namespace kernel_selector
