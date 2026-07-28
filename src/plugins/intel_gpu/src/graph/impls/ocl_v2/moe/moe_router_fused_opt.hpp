// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/primitives/moe_router_fused.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

struct MoeRouterFusedOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::moe::moe_router_fused_opt")
    explicit MoeRouterFusedOpt(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_types = {
            ov::element::f16,
        };

        const auto& in_layout = node.get_input_layout(0);
        if (!one_of(in_layout.data_type, supported_types))
            return false;

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
