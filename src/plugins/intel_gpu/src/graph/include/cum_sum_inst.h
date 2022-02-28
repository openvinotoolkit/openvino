// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/cum_sum.hpp"
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<cum_sum> : public typed_program_node_base<cum_sum> {
    using parent = typed_program_node_base<cum_sum>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    size_t inputs_count() const { return get_dependencies().size(); }
};

using cum_sum_node = typed_program_node<cum_sum>;

template <>
class typed_primitive_inst<cum_sum> : public typed_primitive_inst_base<cum_sum> {
    using parent = typed_primitive_inst_base<cum_sum>;

public:
    static layout calc_output_layout(cum_sum_node const& node);
    static std::string to_string(cum_sum_node const& node);
    typed_primitive_inst(network& network, cum_sum_node const& desc);
};

using cum_sum_inst = typed_primitive_inst<cum_sum>;
}  // namespace cldnn
