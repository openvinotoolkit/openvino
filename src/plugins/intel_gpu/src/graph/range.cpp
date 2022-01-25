// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <range_inst.h>
#include <primitive_type_base.h>
#include "lexical_cast.hpp"

namespace cldnn {

primitive_type_id range::type_id() {
    static primitive_type_base<range> instance;
    return &instance;
}

std::string typed_primitive_inst<range>::to_string(range_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    json_composite op_info;
    op_info.add("output_type", data_type_traits::name(desc->output_layout.data_type));

    node_info->add("range info", std::move(op_info));
    return lexical_cast(*node_info);
}

}  // namespace cldnn
