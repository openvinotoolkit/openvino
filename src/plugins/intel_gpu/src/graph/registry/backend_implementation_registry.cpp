// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend_implementation_registry.hpp"

namespace ov::intel_gpu {

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& backend_implementation_registry::get(std::type_index primitive_type) {
#ifdef OV_GPU_WITH_OPTIONAL_IMPLEMENTATION_PROVIDER
    return backend_extensions::get_compiled_implementations(primitive_type);
#else
    using implementations = std::vector<std::shared_ptr<cldnn::ImplementationManager>>;
    static const implementations empty;
    return empty;
#endif
}

}  // namespace ov::intel_gpu
