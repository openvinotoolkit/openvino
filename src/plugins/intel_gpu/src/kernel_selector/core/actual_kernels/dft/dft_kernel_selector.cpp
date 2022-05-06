// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dft_kernel_selector.h"

#include "dft_kernel_ref.h"

namespace kernel_selector {
namespace {

class dft_kernel_selector : public kernel_selector_base {
    KernelsData GetBestKernels(const Params& params, const optional_params& options) const override {
        return GetNaiveBestKernel(params, options, KernelType::DFT);
    }

public:
    dft_kernel_selector() {
        Attach<DFTKernelRef>();
    }
};

}  // namespace

kernel_selector_base& dft_instance() {
    static dft_kernel_selector instance;
    return instance;
}

}  // namespace kernel_selector
