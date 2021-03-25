// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "input_layout_inst.h"
#include "primitive_type_base.h"
#include "memory_impl.h"
#include "error_handler.h"
#include "json_object.h"
#include <string>
#include <memory>
#include <algorithm>

namespace cldnn {
primitive_type_id input_layout::type_id() {
    static primitive_type_base<input_layout> instance;
    return &instance;
}

input_layout_node::typed_program_node(const std::shared_ptr<input_layout> dprim, program_impl& prog)
    : parent(dprim, prog) {
    can_share_buffer(false);
}

input_layout_inst::typed_primitive_inst(network_impl& network, input_layout_node const& node) : parent(network, node) {
    _has_valid_input = false;  // by default input for 'input_layout' is invalid as long as user doesn't call set_data
}

void input_layout_inst::set_data(memory_impl& mem) {
    auto ol = node.get_output_layout();

    check_memory_to_set(mem, ol);

    if (mem.is_allocated_by(get_network().get_engine())) {
        _output = (memory_impl::ptr) &mem;
    } else {
        mem_lock<char> src((memory_impl::ptr) &mem);
        mem_lock<char> dst(_output);
        std::copy(src.begin(), src.end(), dst.begin());
    }

    _has_valid_input = true;
    _output_changed = true;
}

std::string input_layout_inst::to_string(input_layout_node const& node) {
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    node_info->dump(primitive_description);

    return primitive_description.str();
}

}  // namespace cldnn
