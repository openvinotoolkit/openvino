// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/mvn.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<mvn> : public typed_program_node_base<mvn> {
    using parent = typed_program_node_base<mvn>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
};

using mvn_node = typed_program_node<mvn>;

template <>
class typed_primitive_inst<mvn> : public typed_primitive_inst_base<mvn> {
    using parent = typed_primitive_inst_base<mvn>;

public:
    static layout calc_output_layout(mvn_node const& node);
    static std::string to_string(mvn_node const& node);

public:
    typed_primitive_inst(network& network, mvn_node const& node);
};

using mvn_inst = typed_primitive_inst<mvn>;

}  // namespace cldnn
