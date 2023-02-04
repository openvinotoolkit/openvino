// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/adaptive_pooling.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {

using adaptive_pooling_node = typed_program_node<adaptive_pooling>;

template <>
class typed_primitive_inst<adaptive_pooling> : public typed_primitive_inst_base<adaptive_pooling> {
    using parent = typed_primitive_inst_base<adaptive_pooling>;
    using parent::parent;

public:
    static layout calc_output_layout(const adaptive_pooling_node& node, kernel_impl_params const& impl_param);
    static std::string to_string(const adaptive_pooling_node& node);

    typed_primitive_inst(network& network, const adaptive_pooling_node& node) : parent(network, node) {}
};

using adaptive_pooling_inst = typed_primitive_inst<adaptive_pooling>;

}  // namespace cldnn
