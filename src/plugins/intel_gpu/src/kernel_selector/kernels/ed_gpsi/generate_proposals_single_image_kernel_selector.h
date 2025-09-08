// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class experimental_detectron_generate_proposals_single_image_kernel_selector : public kernel_selector_base {
public:
    static experimental_detectron_generate_proposals_single_image_kernel_selector& Instance();

    experimental_detectron_generate_proposals_single_image_kernel_selector();

    KernelsData GetBestKernels(const Params& params) const override;
};

}  // namespace kernel_selector
