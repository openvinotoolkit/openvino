// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/gather_nd.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {

using gather_nd_node = typed_program_node<gather_nd>;

template <>
class typed_primitive_inst<gather_nd> : public typed_primitive_inst_base<gather_nd> {
    using parent = typed_primitive_inst_base<gather_nd>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(gather_nd_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(gather_nd_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(gather_nd_node const& node);

public:
    typed_primitive_inst(network& network, gather_nd_node const& desc);
};

using gather_nd_inst = typed_primitive_inst<gather_nd>;
}  // namespace cldnn
