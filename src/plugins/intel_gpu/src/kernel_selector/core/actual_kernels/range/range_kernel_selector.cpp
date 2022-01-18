// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "range_kernel_selector.h"
#include "range_kernel_ref.h"

namespace kernel_selector {
namespace {

class range_kernel_selector: public kernel_selector_base {
    KernelsData GetBestKernels(const Params &params, const optional_params &options) const override {
        return GetNaiveBestKernel(params, options, KernelType::RANGE);
    }
public:
    range_kernel_selector() {
        Attach<RangeKernelRef>();
    }
};

}  // namespace

kernel_selector_base& range_instance() {
    static range_kernel_selector instance;
    return instance;
}

}  // namespace kernel_selector
