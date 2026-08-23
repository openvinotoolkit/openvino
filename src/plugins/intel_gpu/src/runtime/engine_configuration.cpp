// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/engine_configuration.hpp"

#include "intel_gpu/runtime/runtime_backend_registry.hpp"

namespace cldnn {

engine_types get_default_engine_type() {
    return runtime_backend_registry::default_backend().engine_type;
}

runtime_types get_default_runtime_type() {
    return runtime_backend_registry::default_backend().runtime_type;
}
}  // namespace cldnn
