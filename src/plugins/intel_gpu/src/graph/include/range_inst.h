// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/range.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<range> : public typed_program_node_base<range> {
private:
    using parent = typed_program_node_base<range>;

public:
    using parent::parent;
    program_node& input(std::size_t i = 0) const { return get_dependency(i); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {0, 1, 2}; }
};

using range_node = typed_program_node<range>;

template <>
class typed_primitive_inst<range> : public typed_primitive_inst_base<range> {
    using parent = typed_primitive_inst_base<range>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(range_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(range_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(range_node const& node);

    typed_primitive_inst(network& network, range_node const& desc);
};

using range_inst = typed_primitive_inst<range>;

}  // namespace cldnn
