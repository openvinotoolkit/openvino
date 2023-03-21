// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <intel_gpu/primitives/eye.hpp>

#include "primitive_inst.h"

namespace cldnn {

using eye_node = typed_program_node<eye>;

template <>
class typed_primitive_inst<eye> : public typed_primitive_inst_base<eye> {
    using parent = typed_primitive_inst_base<eye>;
    using parent::parent;

public:
    static layout calc_output_layout(eye_node const& node, const kernel_impl_params& impl_param);
    static std::string to_string(eye_node const& node);

    typed_primitive_inst(network& network, eye_node const& desc);
};

using eye_inst = typed_primitive_inst<eye>;

}  // namespace cldnn
