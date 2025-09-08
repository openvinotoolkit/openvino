// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <string>

#include "intel_gpu/primitives/reverse.hpp"
#include "primitive_inst.h"

namespace cldnn {

using reverse_node = typed_program_node<reverse>;

template <>
class typed_primitive_inst<reverse> : public typed_primitive_inst_base<reverse> {
    using parent = typed_primitive_inst_base<reverse>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(reverse_node const& /*node*/, const kernel_impl_params& impl_param) {
        return forward_input0_shape<ShapeType>(impl_param);
    }
    static layout calc_output_layout(reverse_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(reverse_node const& node);

    typed_primitive_inst(network& network, reverse_node const& desc);
};

using reverse_inst = typed_primitive_inst<reverse>;
}  // namespace cldnn
