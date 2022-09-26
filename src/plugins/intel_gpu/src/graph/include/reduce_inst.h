// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/reduce.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<reduce> : public typed_program_node_base<reduce> {
    using parent = typed_program_node_base<reduce>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
};

using reduce_node = typed_program_node<reduce>;

template <>
class typed_primitive_inst<reduce> : public typed_primitive_inst_base<reduce> {
    using parent = typed_primitive_inst_base<reduce>;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(reduce_node const& node, const kernel_impl_params& impl_param);
    static layout calc_output_layout(reduce_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(reduce_node const& node);

    typed_primitive_inst(network& network, reduce_node const& desc);
};

using reduce_inst = typed_primitive_inst<reduce>;
}  // namespace cldnn
