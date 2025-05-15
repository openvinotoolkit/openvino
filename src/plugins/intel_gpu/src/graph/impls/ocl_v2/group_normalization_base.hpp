// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

struct GroupNormalizationBase : public ImplementationManager {
    explicit GroupNormalizationBase(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] in_out_fmts_t query_formats(const program_node& node) const override {
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        for (size_t i = 1; i < node.get_dependencies().size(); i++) {
            size_t in_rank = node.get_input_layout(i).get_rank();
            in_fmts[i] = format::get_default_format(in_rank);
        }

        return {in_fmts, out_fmts};
    }
};

}  // namespace ov::intel_gpu::ocl
