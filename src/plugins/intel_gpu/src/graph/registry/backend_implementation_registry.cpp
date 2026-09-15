// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend_implementation_registry.hpp"

#ifndef OV_GPU_WITH_OPTIONAL_IMPLEMENTATION_PROVIDER
namespace ov::intel_gpu::backend_extensions {

const implementations& get_compiled_implementations(std::type_index) {
    static const implementations empty;
    return empty;
}

bool supports_implementation_fusions(cldnn::runtime_types) {
    return true;
}

}  // namespace ov::intel_gpu::backend_extensions
#endif
