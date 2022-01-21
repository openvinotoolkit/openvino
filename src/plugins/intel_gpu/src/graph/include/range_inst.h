// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/range.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<range> : public typed_program_node_base<range> {
    using typed_program_node_base::typed_program_node_base;

    program_node& input(std::size_t i = 0) const { return get_dependency(i); }
};
using range_node = typed_program_node<range>;

template <>
class typed_primitive_inst<range> : public typed_primitive_inst_base<range> {
public:
    static layout calc_output_layout(range_node const& node);
    static std::string to_string(range_node const& node);

    typed_primitive_inst(network& network, range_node const& desc);
};

using range_inst = typed_primitive_inst<range>;

}  // namespace cldnn
