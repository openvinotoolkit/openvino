// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/softmax.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {
using softmax_node = typed_program_node<softmax>;

template <>
class typed_primitive_inst<softmax> : public typed_primitive_inst_base<softmax> {
    using parent = typed_primitive_inst_base<softmax>;

public:
    static layout calc_output_layout(softmax_node const& node);
    static std::string to_string(softmax_node const& node);

public:
    typed_primitive_inst(network_impl& network, softmax_node const& desc);
};

using softmax_inst = typed_primitive_inst<softmax>;

}  // namespace cldnn
