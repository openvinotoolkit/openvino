// Copyright (C) 2021-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dft_inst.h>
#include <primitive_type_base.h>
#include "lexical_cast.hpp"

namespace cldnn {

primitive_type_id dft::type_id() {
    static primitive_type_base<dft> instance;
    return &instance;
}

std::string typed_primitive_inst<dft>::to_string(dft_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    return lexical_cast(*node_info);
}

}  // namespace cldnn
