// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/reorg_yolo.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<reorg_yolo> : public typed_program_node_base<reorg_yolo> {
    using parent = typed_program_node_base<reorg_yolo>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using reorg_yolo_node = typed_program_node<reorg_yolo>;

template <>
class typed_primitive_inst<reorg_yolo> : public typed_primitive_inst_base<reorg_yolo> {
    using parent = typed_primitive_inst_base<reorg_yolo>;
    using parent::parent;

public:
template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(reorg_yolo_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(reorg_yolo_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(reorg_yolo_node const& node);

public:
    typed_primitive_inst(network& network, reorg_yolo_node const& desc);
};

using reorg_yolo_inst = typed_primitive_inst<reorg_yolo>;

}  // namespace cldnn
