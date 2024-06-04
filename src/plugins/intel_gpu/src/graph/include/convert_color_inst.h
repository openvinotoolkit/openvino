// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "intel_gpu/primitives/convert_color.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {
template <>
struct typed_program_node<convert_color> : public typed_program_node_base<convert_color> {
    using parent = typed_program_node_base<convert_color>;

public:
    using parent::parent;
    program_node& input(size_t index = 0) const { return get_dependency(index); }
};

using convert_color_node = typed_program_node<convert_color>;

template <>
class typed_primitive_inst<convert_color> : public typed_primitive_inst_base<convert_color> {
    using parent = typed_primitive_inst_base<convert_color>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(convert_color_node const& /* node */, const kernel_impl_params& impl_param);
    static layout calc_output_layout(convert_color_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(convert_color_node const& node);
    typed_primitive_inst(network& network, convert_color_node const& desc);
};

using convert_color_inst = typed_primitive_inst<convert_color>;
}  // namespace cldnn
