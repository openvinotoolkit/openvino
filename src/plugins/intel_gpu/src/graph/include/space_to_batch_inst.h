// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/space_to_batch.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

using space_to_batch_node = typed_program_node<space_to_batch>;

template <>
class typed_primitive_inst<space_to_batch> : public typed_primitive_inst_base<space_to_batch> {
    using parent = typed_primitive_inst_base<space_to_batch>;
    using parent::parent;

public:
    static layout calc_output_layout(space_to_batch_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(space_to_batch_node const& node);

    typed_primitive_inst(network& network, space_to_batch_node const& desc);
};

using space_to_batch_inst = typed_primitive_inst<space_to_batch>;
}  // namespace cldnn
