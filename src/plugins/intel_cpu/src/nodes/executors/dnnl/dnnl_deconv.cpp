// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_deconv.hpp"
#include "ie_parallel.hpp"
#include <dnnl_extension_utils.h>
#include <memory_desc/cpu_memory_desc_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include <oneapi/dnnl/dnnl.hpp>

namespace ov {
namespace intel_cpu {

//FIXME: add context
DNNLDeconvExecutor::DNNLDeconvExecutor() : DeconvExecutor() {}

bool DNNLDeconvExecutor::init(const DeconvAttrs& deconvAttrs,
                          const std::vector<MemoryDescPtr>& srcDescs,
                          const std::vector<MemoryDescPtr>& dstDescs,
                          const dnnl::primitive_attr &attr) {
}

void DNNLDeconvExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, const void *post_ops_data_) {
}

}   // namespace intel_cpu
}   // namespace ov