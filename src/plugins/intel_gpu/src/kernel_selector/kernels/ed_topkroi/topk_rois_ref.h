// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {

/**
 * ExperimentalDetectronTopKROIs kernel params.
 */
struct experimental_detectron_topk_roi_params : public base_params {
    experimental_detectron_topk_roi_params() : base_params(KernelType::EXPERIMENTAL_DETECTRON_TOPK_ROIS) {}

    size_t max_rois = 0; // maximal numbers of output ROIs.
};

/**
 * Reference GPU kernel for the ExperimentalDetectronTopKROIs-6 operation to set output by indices sorted before.
 */
class ExperimentalDetectronTopKROIRef : public KernelBaseOpenCL {
public:
    ExperimentalDetectronTopKROIRef() : KernelBaseOpenCL("experimental_detectron_topk_rois_ref") {}

private:
    virtual JitConstants GetJitConstants(const experimental_detectron_topk_roi_params &params) const;

    KernelsData GetKernelsData(const Params &params) const override;

    KernelsPriority GetKernelsPriority(const Params &params) const override;

    bool Validate(const Params &params) const override;

    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
