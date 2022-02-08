// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <intel_gpu/primitives/experimental_detectron_prior_grid_generator.hpp>
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<experimental_detectron_prior_grid_generator> : public typed_program_node_base<experimental_detectron_prior_grid_generator> {
    using typed_program_node_base::typed_program_node_base;

    program_node& input(std::size_t i = 0) const { return get_dependency(i); }
};
using experimental_detectron_prior_grid_generator_node = typed_program_node<experimental_detectron_prior_grid_generator>;

template <>
class typed_primitive_inst<experimental_detectron_prior_grid_generator> : public typed_primitive_inst_base<experimental_detectron_prior_grid_generator> {
public:
    using typed_primitive_inst_base::typed_primitive_inst_base;

    static layout calc_output_layout(const experimental_detectron_prior_grid_generator_node &node) {
        return node.get_primitive()->output_layout;
    }
    static std::string to_string(experimental_detectron_prior_grid_generator_node const& node);
};

}  // namespace cldnn
