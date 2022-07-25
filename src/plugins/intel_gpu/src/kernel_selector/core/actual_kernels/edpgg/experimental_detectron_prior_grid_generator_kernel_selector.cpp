// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "experimental_detectron_prior_grid_generator_kernel_selector.h"
#include "experimental_detectron_prior_grid_generator_kernel_ref.h"

namespace kernel_selector {
namespace {

class experimental_detectron_prior_grid_generator_kernel_selector : public kernel_selector_base {
    KernelsData GetBestKernels(const Params &params, const optional_params &options) const override {
        return GetNaiveBestKernel(params, options, KernelType::EXPERIMENTAL_DETECTRON_PRIOR_GRID_GENERATOR);
    }
public:
    experimental_detectron_prior_grid_generator_kernel_selector() {
        Attach<ExperimentalDetectronPriorGridGeneratorKernelRef>();
    }
};

}  // namespace

kernel_selector_base& experimental_detectron_prior_grid_generator_instance() {
    static experimental_detectron_prior_grid_generator_kernel_selector instance;
    return instance;
}

}  // namespace kernel_selector
