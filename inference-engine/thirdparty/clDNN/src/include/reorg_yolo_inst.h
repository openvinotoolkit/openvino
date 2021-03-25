// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/reorg_yolo.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {
using reorg_yolo_node = typed_program_node<reorg_yolo>;

template <>
class typed_primitive_inst<reorg_yolo> : public typed_primitive_inst_base<reorg_yolo> {
    using parent = typed_primitive_inst_base<reorg_yolo>;

public:
    static layout calc_output_layout(reorg_yolo_node const& node);
    static std::string to_string(reorg_yolo_node const& node);

public:
    typed_primitive_inst(network_impl& network, reorg_yolo_node const& desc);
};

using reorg_yolo_inst = typed_primitive_inst<reorg_yolo>;

}  // namespace cldnn
