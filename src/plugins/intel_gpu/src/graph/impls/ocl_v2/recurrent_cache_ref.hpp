// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <utility>

#include "program_node.h"
#include "recurrent_cache_inst.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn; // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct RecurrentCacheRef : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::recurrent_cache::ref")
    explicit RecurrentCacheRef(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node &node,
                                                              const RuntimeParams &params) const override;

    [[nodiscard]] bool validate_impl(const program_node &node) const override {
        assert(node.is_type<recurrent_cache>());
        static constexpr std::array supported_types = {
            ov::element::f16,
            ov::element::f32,
        };

        for (size_t i = 0; i < node.get_dependencies().size(); i++) {
            const auto &in_layout = node.get_input_layout(i);
            const bool data = i == 0 || i == 6 || (node.as<recurrent_cache>().get_primitive()->history < 0 && i == 1);
            if (in_layout.format != format::bfyx ||
                (data ? !one_of(in_layout.data_type, supported_types)
                      : !one_of(in_layout.data_type, {ov::element::i32, ov::element::i64})))
                return false;
        }

        const auto &out_layout = node.get_output_layout(0);
        return out_layout.format == format::bfyx && one_of(out_layout.data_type, supported_types);
    }
};

} // namespace ov::intel_gpu::ocl