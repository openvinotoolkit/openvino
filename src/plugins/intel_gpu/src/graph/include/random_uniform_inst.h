// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/random_uniform.hpp"
#include "primitive_inst.h"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {

using random_uniform_node = typed_program_node<random_uniform>;

template<>
class typed_primitive_inst<random_uniform> : public typed_primitive_inst_base<random_uniform> {
    using parent = typed_primitive_inst_base<random_uniform>;
    using parent::parent;

public:
    static layout calc_output_layout(random_uniform_node const &node, kernel_impl_params const& impl_param);

    static std::string to_string(random_uniform_node const &node);

public:
    typed_primitive_inst(network &network, random_uniform_node const &desc);
};

using random_uniform_inst = typed_primitive_inst<random_uniform>;

} // namespace cldnn
