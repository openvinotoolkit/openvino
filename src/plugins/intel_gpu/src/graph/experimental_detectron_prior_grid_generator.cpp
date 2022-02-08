// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <experimental_detectron_prior_grid_generator_inst.h>
#include <primitive_type_base.h>
#include "lexical_cast.hpp"

namespace cldnn {

primitive_type_id experimental_detectron_prior_grid_generator::type_id() {
    static primitive_type_base<experimental_detectron_prior_grid_generator> instance;
    return &instance;
}

std::string typed_primitive_inst<experimental_detectron_prior_grid_generator>::to_string(experimental_detectron_prior_grid_generator_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    return lexical_cast(*node_info);
}

}  // namespace cldnn
