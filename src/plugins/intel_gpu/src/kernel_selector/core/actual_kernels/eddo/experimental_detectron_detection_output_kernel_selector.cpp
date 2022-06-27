// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "experimental_detectron_detection_output_kernel_selector.h"

#include "experimental_detectron_detection_output_kernel_ref.h"

namespace kernel_selector {
experimental_detectron_detection_output_kernel_selector::experimental_detectron_detection_output_kernel_selector() {
    Attach<ExperimentalDetectronDetectionOutputKernelRef>();
}

experimental_detectron_detection_output_kernel_selector&
experimental_detectron_detection_output_kernel_selector::Instance() {
    static experimental_detectron_detection_output_kernel_selector instance_;
    return instance_;
}

KernelsData experimental_detectron_detection_output_kernel_selector::GetBestKernels(
    const Params& params,
    const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::EXPERIMENTAL_DETECTRON_DETECTION_OUTPUT);
}
}  // namespace kernel_selector
