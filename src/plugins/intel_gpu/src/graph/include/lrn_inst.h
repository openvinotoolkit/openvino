// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/lrn.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

using lrn_node = typed_program_node<lrn>;

template <>
class typed_primitive_inst<lrn> : public typed_primitive_inst_base<lrn> {
    using parent = typed_primitive_inst_base<lrn>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(lrn_node const& /*node*/, const kernel_impl_params& impl_param) {
        return forward_input0_shape<ShapeType>(impl_param);
    }
    static layout calc_output_layout(lrn_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(lrn_node const& node);

    typed_primitive_inst(network& network, lrn_node const& node);
};

using lrn_inst = typed_primitive_inst<lrn>;

}  // namespace cldnn
