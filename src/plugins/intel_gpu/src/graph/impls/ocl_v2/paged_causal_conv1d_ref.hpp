// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "paged_causal_conv1d_inst.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct PagedCausalConv1DRef : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::paged_causal_conv1d::ref")
    explicit PagedCausalConv1DRef(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<paged_causal_conv1d>());
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

            if (i <= paged_causal_conv1d::CONV_BIAS) {
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

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
