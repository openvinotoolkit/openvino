// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/grn.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

using grn_node = typed_program_node<grn>;

template <>
class typed_primitive_inst<grn> : public typed_primitive_inst_base<grn> {
    using parent = typed_primitive_inst_base<grn>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(grn_node const& /*node*/, const kernel_impl_params& impl_param) {
        return forward_input0_shape<ShapeType>(impl_param);
    }
    static layout calc_output_layout(grn_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(grn_node const& node);

    typed_primitive_inst(network& network, grn_node const& node);
};

using grn_inst = typed_primitive_inst<grn>;

}  // namespace cldnn
