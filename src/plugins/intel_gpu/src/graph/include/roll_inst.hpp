// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/roll.hpp"
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<roll> : public typed_program_node_base<roll> {
    using parent = typed_program_node_base<roll>;

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using roll_node = typed_program_node<roll>;

template <>
class typed_primitive_inst<roll> : public typed_primitive_inst_base<roll> {
public:
    using parent = typed_primitive_inst_base<roll>;
    using parent::parent;

    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(roll_node const& /*node*/, const kernel_impl_params& impl_param) {
        return forward_input0_shape<ShapeType>(impl_param);
    }
    static layout calc_output_layout(const roll_node& node, kernel_impl_params const& impl_param);
    static std::string to_string(const roll_node& node);
};

using roll_inst = typed_primitive_inst<roll>;

}  // namespace cldnn
