// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/gather_elements.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {
template <>
struct typed_program_node<gather_elements> : public typed_program_node_base<gather_elements> {
    using parent = typed_program_node_base<gather_elements>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using gather_elements_node = typed_program_node<gather_elements>;

template <>
class typed_primitive_inst<gather_elements> : public typed_primitive_inst_base<gather_elements> {
    using parent = typed_primitive_inst_base<gather_elements>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(gather_elements_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(gather_elements_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(gather_elements_node const& node);

public:
    typed_primitive_inst(network& network, gather_elements_node const& desc);
};

using gather_elements_inst = typed_primitive_inst<gather_elements>;
}  // namespace cldnn
