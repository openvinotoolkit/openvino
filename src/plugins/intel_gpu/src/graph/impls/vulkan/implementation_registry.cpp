// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <unordered_map>

#include "eltwise.hpp"
#include "registry/backend_implementation_registry.hpp"
#include "registry/predicates.hpp"
#include "reorder.hpp"
#include "reshape.hpp"

namespace ov::intel_gpu::backend_extensions {

const implementations& get_compiled_implementations(std::type_index primitive_type) {
    static const implementations empty;
    static const std::unordered_map<std::type_index, implementations> compiled_implementations = {
        {typeid(cldnn::eltwise),
         {std::make_shared<cldnn::vulkan::EltwiseImplementationManager>(cldnn::shape_types::static_shape, cldnn::not_in_shape_flow()),
          std::make_shared<cldnn::vulkan::EltwiseImplementationManager>(cldnn::shape_types::dynamic_shape, cldnn::not_in_shape_flow())}},
        {typeid(cldnn::reorder),
         {std::make_shared<cldnn::vulkan::ReorderImplementationManager>(cldnn::shape_types::static_shape, cldnn::not_in_shape_flow()),
          std::make_shared<cldnn::vulkan::ReorderImplementationManager>(cldnn::shape_types::dynamic_shape, cldnn::not_in_shape_flow())}},
        {typeid(cldnn::reshape), {std::make_shared<cldnn::vulkan::ReshapeImplementationManager>(cldnn::shape_types::static_shape)}},
    };

    const auto implementation = compiled_implementations.find(primitive_type);
    return implementation == compiled_implementations.end() ? empty : implementation->second;
}

}  // namespace ov::intel_gpu::backend_extensions
