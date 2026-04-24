// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "intel_gpu/primitives/scaled_shifted_clamp_experimental.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<scaled_shifted_clamp_experimental>
    : public typed_program_node_base<scaled_shifted_clamp_experimental> {
    using parent = typed_program_node_base<scaled_shifted_clamp_experimental>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using scaled_shifted_clamp_experimental_node = typed_program_node<scaled_shifted_clamp_experimental>;

template <>
class typed_primitive_inst<scaled_shifted_clamp_experimental>
    : public typed_primitive_inst_base<scaled_shifted_clamp_experimental> {
    using parent = typed_primitive_inst_base<scaled_shifted_clamp_experimental>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(scaled_shifted_clamp_experimental_node const& node,
                                                   const kernel_impl_params& impl_params);
    static layout calc_output_layout(scaled_shifted_clamp_experimental_node const& node,
                                     kernel_impl_params const& impl_params);
    static std::string to_string(scaled_shifted_clamp_experimental_node const& node);

    typed_primitive_inst(network& network, scaled_shifted_clamp_experimental_node const& node);
};

using scaled_shifted_clamp_experimental_inst = typed_primitive_inst<scaled_shifted_clamp_experimental>;

}  // namespace cldnn
