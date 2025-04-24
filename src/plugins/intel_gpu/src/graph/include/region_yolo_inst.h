// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/region_yolo.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<region_yolo> : public typed_program_node_base<region_yolo> {
    using parent = typed_program_node_base<region_yolo>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using region_yolo_node = typed_program_node<region_yolo>;

template <>
class typed_primitive_inst<region_yolo> : public typed_primitive_inst_base<region_yolo> {
    using parent = typed_primitive_inst_base<region_yolo>;
    using parent::parent;

public:
template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(region_yolo_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(region_yolo_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(region_yolo_node const& node);

public:
    typed_primitive_inst(network& network, region_yolo_node const& desc);
};

using region_yolo_inst = typed_primitive_inst<region_yolo>;

}  // namespace cldnn
