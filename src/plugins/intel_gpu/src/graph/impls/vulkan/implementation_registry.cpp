// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <unordered_map>

#include "backend_graph_optimizer.hpp"
#include "eltwise.hpp"
#include "registry/backend_implementation_registry.hpp"
#include "registry/predicates.hpp"
#include "reorder.hpp"
#include "reshape.hpp"

namespace ov::intel_gpu::backend_extensions {
namespace {

class reference_kernel_graph_optimizer final : public cldnn::backend_graph_optimizer {
public:
    bool optimize_fusions(cldnn::program&) const override {
        // Reference adapters accept unfused primitives. Keep common graph cleanup,
        // but do not apply fusions whose implementation belongs to another backend.
        return true;
    }
};

}  // namespace

const implementations& get_compiled_implementations(std::type_index primitive_type) {
    static const reference_kernel_graph_optimizer graph_optimizer;
    static const cldnn::backend_graph_optimizer_registration graph_optimizer_registration{cldnn::runtime_types::vulkan, graph_optimizer};
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
