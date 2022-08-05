// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/lrn.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

using lrn_node = typed_program_node<lrn>;

template <>
class typed_primitive_inst<lrn> : public typed_primitive_inst_base<lrn> {
    using parent = typed_primitive_inst_base<lrn>;

public:
    static layout calc_output_layout(lrn_node const& node);
    static std::string to_string(lrn_node const& node);

public:
    typed_primitive_inst(network& network, lrn_node const& node);
};

using lrn_inst = typed_primitive_inst<lrn>;

}  // namespace cldnn
