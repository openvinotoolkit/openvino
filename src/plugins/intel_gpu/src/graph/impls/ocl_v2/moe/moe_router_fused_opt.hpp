// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "intel_gpu/primitives/moe_router_fused.hpp"
#include "primitive_inst.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

using RouterPredictionEntry = std::function<cldnn::event::ptr(
    const std::vector<cldnn::memory::ptr>&,
    const std::vector<cldnn::memory::ptr>&,
    const std::vector<cldnn::event::ptr>&)>;

RouterPredictionEntry make_router_prediction_entry(cldnn::primitive_inst& instance);

struct MoeRouterFusedOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::moe::moe_router_fused_opt")
    explicit MoeRouterFusedOpt(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_types = {
            ov::element::f16,
        };

        const auto& in_layout = node.get_input_layout(0);
        return one_of(in_layout.data_type, supported_types);
    }
};

}  // namespace ov::intel_gpu::ocl
