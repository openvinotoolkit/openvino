// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/generic_primitive.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<generic_primitive> : public typed_program_node_base<generic_primitive> {
    using parent = typed_program_node_base<generic_primitive>;

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
};

using generic_primitive_node = typed_program_node<generic_primitive>;

template <>
class typed_primitive_inst<generic_primitive> : public typed_primitive_inst_base<generic_primitive> {
    using parent = typed_primitive_inst_base<generic_primitive>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(generic_primitive_node const& node, const kernel_impl_params& impl_param);
    static layout calc_output_layout(generic_primitive_node const& node, kernel_impl_params const& impl_param);

    static std::string to_string(generic_primitive_node const& node);

    typed_primitive_inst(network& network, generic_primitive_node const& node);
};

using generic_primitive_inst = typed_primitive_inst<generic_primitive>;

}  // namespace cldnn
