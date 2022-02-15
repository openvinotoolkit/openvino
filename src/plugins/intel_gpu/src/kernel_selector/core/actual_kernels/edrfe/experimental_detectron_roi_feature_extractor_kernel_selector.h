// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class experimental_detectron_roi_feature_extractor_kernel_selector : public kernel_selector_base {
public:
    static experimental_detectron_roi_feature_extractor_kernel_selector& Instance();

    experimental_detectron_roi_feature_extractor_kernel_selector();
    ~experimental_detectron_roi_feature_extractor_kernel_selector() = default;

    KernelsData GetBestKernels(const Params& params, const optional_params& options) const override;
};
}  // namespace kernel_selector
