// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/split.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

using split_node = typed_program_node<split>;

template <>
class typed_primitive_inst<split> : public typed_primitive_inst_base<split> {
    using parent = typed_primitive_inst_base<split>;
    using parent::parent;

public:
    static layout calc_output_layout(split_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(split_node const& node);
    typed_primitive_inst(network& network, split_node const& node);
};

using split_inst = typed_primitive_inst<split>;
}  // namespace cldnn
