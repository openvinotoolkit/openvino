// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common_utils/kernels_cache.hpp"
#include "intel_gpu/runtime/kernel_builder.hpp"

namespace cldnn {

struct kernel_cache_frontend_context {
    size_t max_kernels_per_batch = 1;
    uint32_t program_id = 0;
    std::string device_name;
    std::string driver_version;
    std::string dump_sources_path;
    const std::map<std::string, std::string>* batch_headers = nullptr;
    KernelSourceHeaders source_headers = KernelSourceHeaders::BATCH_PREAMBLE;
    std::string compiler_cache_identity;
};

/// Converts pending kernel descriptions into compiler/artifact batches.
///
/// Source preprocessing is isolated behind an OCLC/CM policy. Precompiled
/// SPIR-V takes the artifact path and never invokes include expansion, option
/// normalization, or OpenCL source-dump policy.
class kernel_cache_frontend final {
public:
    static void prepare(const kernels_cache::kernels_code& pending,
                        const kernel_cache_frontend_context& context,
                        std::vector<kernels_cache::batch_program>& batches);
};

}  // namespace cldnn
