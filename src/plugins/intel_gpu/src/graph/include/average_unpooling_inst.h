// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/average_unpooling.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

using average_unpooling_node = typed_program_node<average_unpooling>;

template <>
class typed_primitive_inst<average_unpooling> : public typed_primitive_inst_base<average_unpooling> {
    using parent = typed_primitive_inst_base<average_unpooling>;
    using parent::parent;

public:
    typed_primitive_inst(network& network, average_unpooling_node const& desc);
    static layout calc_output_layout(average_unpooling_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(average_unpooling_node const& node);
};

using average_unpooling_inst = typed_primitive_inst<average_unpooling>;

}  // namespace cldnn
