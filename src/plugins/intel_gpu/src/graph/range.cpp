// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "range_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
namespace {
std::string lexical_cast(const json_base& j, int offset = 1) {
    std::stringstream os;
    j.dump(os, offset);
    return os.str();
}
}  // namespace

primitive_type_id range::type_id() {
    static primitive_type_base<range> instance;
    return &instance;
}

layout range_inst::calc_output_layout(range_node const& node) {
    return node.get_primitive()->output_layout;
}

std::string range_inst::to_string(range_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite op_info;
    op_info.add("output_type", data_type_traits::name(desc->output_layout.data_type));

    node_info->add("range info", std::move(op_info));
    return lexical_cast(*node_info);
}

range_inst::typed_primitive_inst(network& network, range_node const& node) : typed_primitive_inst_base{network, node} {}

}  // namespace cldnn
