// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

// TODO: need confirm, gate is 1st matmul or up is 1st matmul?
//  mlp_gate: 0
//  mlp_up: 1
//  mlp_down: 2
enum class MOEInputIndex : uint8_t {
    HIDDEN_STATES = 0,
    ROUTING_WEIGHTS = 1,
    ROUTER_TOPK_OUTPUT_INDICES = 2,
    WEIGHT_0 = 3,
    SCALE_0 = 4,
    ZP_0 = 5,
    WEIGHT_1 = 6,
    SCALE_1 = 7,
    ZP_1 = 8,
    WEIGHT_2 = 9,
    SCALE_2 = 10,
    ZP_2 = 11
};

struct MOEOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::moe::opt")
    explicit MOEOpt(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_fmts = {
            format::bfyx,
        };

        // TODO(MOE): support more precision
        static constexpr std::array supported_types = {
            ov::element::f16,
        };

        const auto& in0_layout = node.get_input_layout(static_cast<size_t>(MOEInputIndex::HIDDEN_STATES));
        const auto& out_layout = node.get_output_layout(0);
        if (!one_of(in0_layout.format, supported_fmts) || !one_of(out_layout.format, supported_fmts)) {
            return false;
        }

        if (!one_of(in0_layout.data_type, supported_types) || !one_of(out_layout.data_type, supported_types)) {
            return false;
        }

        // Only support u4 weights for now
        static constexpr std::array supported_wei_type = {
            ov::element::u4,
        };
        const auto& wei_layout = node.get_input_layout(static_cast<size_t>(MOEInputIndex::WEIGHT_0));
        if (!one_of(wei_layout.data_type, supported_wei_type)) {
            return false;
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
