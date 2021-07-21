// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "generic_primitive_inst.h"
#include "primitive_type_base.h"
#include <sstream>
#include "json_object.h"
#include <string>

namespace cldnn {

    primitive_type_id generic_primitive::type_id() {
        static primitive_type_base<generic_primitive> instance;
        return &instance;
    }

    std::string generic_primitive_inst::to_string(generic_primitive_node const& node) {
        auto desc = node.get_primitive();
        auto node_info = node.desc_to_json();

        std::stringstream primitive_description;

        json_composite generic_prim_info;
        // TODO: consider printing more information here
        node_info->add("custom primitive info", generic_prim_info);
        node_info->dump(primitive_description);

        return primitive_description.str();
    }

    generic_primitive_inst::typed_primitive_inst(network_impl& network, generic_primitive_node const& node)
    : parent(network, node) {}
}  // namespace cldnn
