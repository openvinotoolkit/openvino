// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/select.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<select> : public typed_program_node_base<select> {
    using parent = typed_program_node_base<select>;

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using select_node = typed_program_node<select>;

template <>
class typed_primitive_inst<select> : public typed_primitive_inst_base<select> {
    using parent = typed_primitive_inst_base<select>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(const select_node& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(select_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(select_node const& node);
    typed_primitive_inst(network& network, select_node const& node);
};

using select_inst = typed_primitive_inst<select>;
}  // namespace cldnn
