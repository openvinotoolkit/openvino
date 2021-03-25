// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/region_yolo.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {
using region_yolo_node = typed_program_node<region_yolo>;

template <>
class typed_primitive_inst<region_yolo> : public typed_primitive_inst_base<region_yolo> {
    using parent = typed_primitive_inst_base<region_yolo>;

public:
    static layout calc_output_layout(region_yolo_node const& node);
    static std::string to_string(region_yolo_node const& node);

public:
    typed_primitive_inst(network_impl& network, region_yolo_node const& desc);
};

using region_yolo_inst = typed_primitive_inst<region_yolo>;

}  // namespace cldnn
