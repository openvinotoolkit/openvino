// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/grn.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<grn> : public typed_program_node_base<grn> {
    using parent = typed_program_node_base<grn>;

public:
    using parent::parent;

    program_node& input() const { return *get_dependency(0).first; }
};

using grn_node = typed_program_node<grn>;

template <>
class typed_primitive_inst<grn> : public typed_primitive_inst_base<grn> {
    using parent = typed_primitive_inst_base<grn>;

public:
    static layout calc_output_layout(grn_node const& node);
    static std::string to_string(grn_node const& node);

public:
    typed_primitive_inst(network& network, grn_node const& node);
};

using grn_inst = typed_primitive_inst<grn>;

}  // namespace cldnn
