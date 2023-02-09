// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/roll.hpp"
#include "primitive_inst.h"

namespace cldnn {

using roll_node = typed_program_node<roll>;

template <>
class typed_primitive_inst<roll> : public typed_primitive_inst_base<roll> {
public:
    using parent = typed_primitive_inst_base<roll>;
    using parent::parent;

    static layout calc_output_layout(const roll_node& node, kernel_impl_params const& impl_param);
    static std::string to_string(const roll_node& node);
};

using roll_inst = typed_primitive_inst<roll>;

}  // namespace cldnn
