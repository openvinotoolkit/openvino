// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/scatter_nd_update.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {
template <>
struct typed_program_node<scatter_nd_update> : public typed_program_node_base<scatter_nd_update> {
    using parent = typed_program_node_base<scatter_nd_update>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
};

using scatter_nd_update_node = typed_program_node<scatter_nd_update>;

template <>
class typed_primitive_inst<scatter_nd_update> : public typed_primitive_inst_base<scatter_nd_update> {
    using parent = typed_primitive_inst_base<scatter_nd_update>;

public:
    static layout calc_output_layout(scatter_nd_update_node const& node);
    static std::string to_string(scatter_nd_update_node const& node);

public:
    typed_primitive_inst(network& network, scatter_nd_update_node const& desc);
};

using scatter_nd_update_inst = typed_primitive_inst<scatter_nd_update>;
}  // namespace cldnn
