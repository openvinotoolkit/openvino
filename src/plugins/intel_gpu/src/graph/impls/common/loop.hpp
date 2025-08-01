// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry/implementation_manager.hpp"
#include "program_node.h"

#include <memory>

namespace cldnn {

namespace common {

struct LoopImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("common::loop")
    LoopImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::common, shape_type, vf) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    in_out_fmts_t query_formats(const program_node& node) const override {
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);

        for (size_t i = 0; i < node.get_dependencies().size(); i++) {
            size_t in_rank = node.get_input_layout(i).get_rank();
            in_fmts[i] = format::get_default_format(in_rank);
        }
        size_t out_rank = node.get_output_layout().get_rank();
        out_fmts[0] = format::get_default_format(out_rank);

        return {in_fmts, out_fmts};
    }
};

} // namespace common
} // namespace cldnn
