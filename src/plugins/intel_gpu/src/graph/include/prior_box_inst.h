// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/prior_box.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<prior_box> : typed_program_node_base<prior_box> {
    using parent = typed_program_node_base<prior_box>;

    typed_program_node(std::shared_ptr<prior_box> prim, program& prog);

    program_node& input() const { return get_dependency(0); }

    bool is_clustered() const { return get_primitive()->is_clustered(); }
    void calc_result();
    memory::ptr get_result_buffer() const { return result; }

private:
    memory::ptr result;
};

using prior_box_node = typed_program_node<prior_box>;

template <>
class typed_primitive_inst<prior_box> : public typed_primitive_inst_base<prior_box> {
    using parent = typed_primitive_inst_base<prior_box>;

public:
    static layout calc_output_layout(prior_box_node const& node);
    static std::string to_string(prior_box_node const& node);

public:
    typed_primitive_inst(network& network, prior_box_node const& node);

    memory& input_memory() const { return dep_memory(0); }
};

using prior_box_inst = typed_primitive_inst<prior_box>;

}  // namespace cldnn
