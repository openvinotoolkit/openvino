// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <initializer_list>
#include <memory>
#include <typeindex>
#include <vector>

#include "implementation_manager.hpp"
#include "intel_gpu/runtime/engine_configuration.hpp"

namespace ov::intel_gpu {

namespace backend_extensions {

using implementations = std::vector<std::shared_ptr<cldnn::ImplementationManager>>;

/// Implemented by the optional backend selected at build time.
const implementations& get_compiled_implementations(std::type_index primitive_type);
bool supports_implementation_fusions(cldnn::runtime_types runtime);

}  // namespace backend_extensions

template <typename Primitive>
std::vector<std::shared_ptr<cldnn::ImplementationManager>> compose_backend_implementations(
    std::initializer_list<std::shared_ptr<cldnn::ImplementationManager>> common_implementations) {
    const auto& backend_implementations = backend_extensions::get_compiled_implementations(typeid(Primitive));
    std::vector<std::shared_ptr<cldnn::ImplementationManager>> result(backend_implementations.begin(), backend_implementations.end());
    result.insert(result.end(), common_implementations.begin(), common_implementations.end());
    return result;
}

}  // namespace ov::intel_gpu
