// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "hyper_connection_pre_inst.h"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct HyperConnectionPreOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::hyper_connection_pre::opt")
    explicit HyperConnectionPreOpt(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<hyper_connection_pre>());
        // the mix stage maps one 16-lane subgroup to the 4x4 comb matrix
        if (node.as<hyper_connection_pre>().get_primitive()->n_streams != 4) {
            return false;
        }
        static constexpr std::array supported_types = {ov::element::f16, ov::element::f32};
        for (size_t i = 0; i < node.get_dependencies().size(); i++) {
            const auto& l = node.get_input_layout(i);
            // the kernels index rows densely
            if (!one_of(l.data_type, supported_types) || !format::is_default_format(l.format) || l.data_padding) {
                return false;
            }
        }
        for (size_t i = 0; i < node.get_outputs_count(); i++) {
            const auto& l = node.get_output_layout(i);
            if (!one_of(l.data_type, supported_types) || !format::is_default_format(l.format)) {
                return false;
            }
        }
        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
