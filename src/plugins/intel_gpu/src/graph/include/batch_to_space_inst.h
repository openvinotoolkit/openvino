// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/batch_to_space.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

using batch_to_space_node = typed_program_node<batch_to_space>;

template <>
class typed_primitive_inst<batch_to_space> : public typed_primitive_inst_base<batch_to_space> {
    using parent = typed_primitive_inst_base<batch_to_space>;
    using parent::parent;

public:
    static layout calc_output_layout(batch_to_space_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(batch_to_space_node const& node);
    typed_primitive_inst(network& network, batch_to_space_node const& desc);
};

using batch_to_space_inst = typed_primitive_inst<batch_to_space>;
}  // namespace cldnn
