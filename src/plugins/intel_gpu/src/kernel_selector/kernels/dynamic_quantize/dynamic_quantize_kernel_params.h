// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "intel_gpu/primitives/dynamic_quantize.hpp"

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// dynamic_quantize_fuse_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct dynamic_quantize_fuse_params : fuse_params {
    dynamic_quantize_fuse_params(const cldnn::dynamic_quantize::Attributes& attrs, size_t input_size)
        : fuse_params(KernelType::DYNAMIC_QUANTIZE),
          attrs(attrs),
          input_size(input_size) {}
    cldnn::dynamic_quantize::Attributes attrs;
    size_t input_size;
};

}  // namespace kernel_selector
