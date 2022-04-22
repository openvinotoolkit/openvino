// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "intel_gpu/primitives/broadcast.hpp"

#include "primitive_inst.h"
#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<broadcast> : typed_program_node_base<broadcast> {
private:
    using parent = typed_program_node_base<broadcast>;

public:
    using parent::parent;

    typed_program_node(const std::shared_ptr<broadcast> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }
    program_node& input() const { return *get_dependency(0).first; }
};

using broadcast_node = typed_program_node<broadcast>;

template <>
class typed_primitive_inst<broadcast> : public typed_primitive_inst_base<broadcast> {
    using parent = typed_primitive_inst_base<broadcast>;

public:
    static layout calc_output_layout(broadcast_node const& node);
    static std::string to_string(broadcast_node const& node);
    typed_primitive_inst(network& network, broadcast_node const& node);
};

using broadcast_inst = typed_primitive_inst<broadcast>;
}  // namespace cldnn
