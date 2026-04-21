// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include <utility>

#include "paged_gated_delta_net_inst.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct PagedGatedDeltaNetBase : public ImplementationManager {
    explicit PagedGatedDeltaNetBase(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<paged_gated_delta_net>());
        static constexpr std::array supported_fmts = {
            format::bfyx,
        };

        static constexpr std::array supported_real_types = {
            ov::element::f16,
            ov::element::f32,
        };

        for (size_t i = 0; i < node.get_dependencies().size(); i++) {
            const auto& in_layout = node.get_input_layout(i);
            if (!one_of(in_layout.format, supported_fmts)) {
                return false;
            }

            if (i <= paged_gated_delta_net::BETA) {
                if (!one_of(in_layout.data_type, supported_real_types)) {
                    return false;
                }
            } else if (in_layout.data_type != ov::element::i32) {
                return false;
            }
        }

        const auto& out_layout = node.get_output_layout(0);
        if (!one_of(out_layout.format, supported_fmts) || !one_of(out_layout.data_type, supported_real_types)) {
            return false;
        }
        const auto& q_shape = node.get_input_layout(paged_gated_delta_net::QUERY).get_partial_shape();
        const auto& v_shape = node.get_input_layout(paged_gated_delta_net::VALUE).get_partial_shape();
        if (q_shape.rank().is_dynamic() || v_shape.rank().is_dynamic() || q_shape.rank().get_length() < 3 || v_shape.rank().get_length() < 3 ||
            q_shape[2].is_dynamic() || v_shape[2].is_dynamic()) {
            return false;
        }
        return validate_internal(node);
    }

protected:
    virtual bool validate_internal(const program_node& node) const {
        return true;
    }
};

struct PagedGatedDeltaNetRef : public PagedGatedDeltaNetBase {
    OV_GPU_PRIMITIVE_IMPL("ocl::paged_gated_delta_net::ref")
    explicit PagedGatedDeltaNetRef(shape_types shape_type, ValidateFunc vf = nullptr) : PagedGatedDeltaNetBase(shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
};

struct PagedGatedDeltaNetOpt : public PagedGatedDeltaNetBase {
    OV_GPU_PRIMITIVE_IMPL("ocl::paged_gated_delta_net::opt")
    explicit PagedGatedDeltaNetOpt(shape_types shape_type, ValidateFunc vf = nullptr) : PagedGatedDeltaNetBase(shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

protected:
    bool validate_internal(const program_node& node) const override {
        const auto& q_shape = node.get_input_layout(paged_gated_delta_net::QUERY).get_partial_shape();
        const auto& v_shape = node.get_input_layout(paged_gated_delta_net::VALUE).get_partial_shape();

        const auto k_head_dims = q_shape[2].get_length();
        const auto v_head_dims = v_shape[2].get_length();
        return (k_head_dims % 16) == 0 && (v_head_dims % 16) == 0;
    }
};

}  // namespace ov::intel_gpu::ocl
