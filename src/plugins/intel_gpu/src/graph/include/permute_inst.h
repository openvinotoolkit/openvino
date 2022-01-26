// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/permute.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<permute> : public typed_program_node_base<permute> {
    using parent = typed_program_node_base<permute>;
    typed_program_node(const std::shared_ptr<permute> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
};

using permute_node = typed_program_node<permute>;

template <>
class typed_primitive_inst<permute> : public typed_primitive_inst_base<permute> {
    using parent = typed_primitive_inst_base<permute>;

public:
    static layout calc_output_layout(permute_node const& node);
    static std::string to_string(permute_node const& node);

public:
    typed_primitive_inst(network& network, permute_node const& node);
};

using permute_inst = typed_primitive_inst<permute>;

}  // namespace cldnn
