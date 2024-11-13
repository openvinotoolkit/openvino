// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels_db.hpp"
#include <assert.h>
#include <algorithm>
#include <vector>
#include <utility>
#include "openvino/core/except.hpp"

#ifndef NDEBUG
#include <fstream>
#include <iostream>
#endif

namespace ov {
namespace intel_gpu {
namespace ocl {

KernelsDB::KernelsDB()
    // :
//     primitives({
// #include "ks_primitive_db.inc"
//       }),
//       batch_headers({
// #include "ks_primitive_db_batch_headers.inc"
//       }
    //   )
    {
        m_kernels = {
            {"sdpa_ref", {{
            (std::string) R"__krnl(
            this is the kernels code
            #ifdef APPLY_SCALE_TO_QUERY
            #undef APPLY_SCALE_TO_QUERY
            #endif
            #ifdef HAS_KV_CACHE_ZP_INPUT
            #undef HAS_KV_CACHE_ZP_INPUT
            #endif
            #ifdef GET_COMPRESSION_INDEX
            #undef GET_COMPRESSION_INDEX
            #endif
            #ifdef GET_COMPRESSION_INDEX
            #undef GET_COMPRESSION_INDEX
            #endif
            )__krnl"}, {}} },
        };
    }

const KernelTemplateDesc& KernelsDB::get_template(const KernelTemplateID& id) const {
    try {
        auto codes = m_kernels.find(id);
        OPENVINO_ASSERT(codes != m_kernels.end(), "[GPU] Cannot find the kernel " + id + " in kernels database.");
        return codes->second;
    } catch (...) {
        OPENVINO_THROW("[GPU] Cannot find the kernel " + id + " in kernels database.");
    }
}

}  // namespace ocl
}  // namespace intel_gpu
}  // namespace ov
