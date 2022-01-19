// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "experimental_detectron_roi_feature_extractor_kernel_selector.h"
#include "experimental_detectron_roi_feature_extractor_kernel_ref.h"

namespace kernel_selector {

experimental_detectron_roi_feature_extractor_kernel_selector& experimental_detectron_roi_feature_extractor_kernel_selector::Instance() {
        static experimental_detectron_roi_feature_extractor_kernel_selector instance_;
        return instance_;
    }

experimental_detectron_roi_feature_extractor_kernel_selector::experimental_detectron_roi_feature_extractor_kernel_selector() {
    implementations.push_back(std::make_shared<ExperimentalDetectronROIFeatureExtractorRef>());
}

KernelsData experimental_detectron_roi_feature_extractor_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::EXPERIMENTAL_DETECTRON_ROI_FEATURE_EXTRACTOR);
}

}  // namespace kernel_selector
