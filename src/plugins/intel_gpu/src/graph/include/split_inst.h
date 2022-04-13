// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "intel_gpu/primitives/split.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
class typed_program_node<split> : public typed_program_node_base<split> {
    using parent = typed_program_node_base<split>;

public:
    using parent::parent;

    program_node& input() const { return *get_dependency(0).first; }
};

using split_node = typed_program_node<split>;

template <>
class typed_primitive_inst<split> : public typed_primitive_inst_base<split> {
    using parent = typed_primitive_inst_base<split>;

public:
    static layout calc_output_layout(split_node const& node);
    static std::string to_string(split_node const& node);
    typed_primitive_inst(network& network, split_node const& node);
};

using split_inst = typed_primitive_inst<split>;
}  // namespace cldnn
