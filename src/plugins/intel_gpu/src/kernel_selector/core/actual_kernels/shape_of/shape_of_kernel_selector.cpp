// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_of_kernel_selector.h"
#include "shape_of_kernel_ref.h"

namespace kernel_selector {
namespace {

class shape_of_kernel_selector: public kernel_selector_base {
    KernelsData GetBestKernels(const Params &params, const optional_params &options) const override {
        return GetNaiveBestKernel(params, options, KernelType::SHAPE_OF);
    }
public:
    shape_of_kernel_selector() {
        Attach<ShapeOfKernelRef>();
    }
};

}  // namespace

kernel_selector_base& shape_of_instance() {
    static shape_of_kernel_selector instance;
    return instance;
}

}  // namespace kernel_selector
