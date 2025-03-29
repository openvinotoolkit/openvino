// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/detection_output.hpp"
#include "primitive_inst.h"

#include <string>

#define PRIOR_BOX_SIZE 4  // Each prior-box consists of [xmin, ymin, xmax, ymax].
#define DETECTION_OUTPUT_ROW_SIZE \
    (3 + PRIOR_BOX_SIZE)  // Each detection consists of [image_id, label, confidence, xmin, ymin, xmax, ymax].

namespace cldnn {

template <>
class typed_program_node<detection_output> : public typed_program_node_base<detection_output> {
    using parent = typed_program_node_base<detection_output>;

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& location() const { return get_dependency(0); }
    program_node& confidence() const { return get_dependency(1); }
    program_node& prior_box() const { return get_dependency(2); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using detection_output_node = typed_program_node<detection_output>;

template <>
class typed_primitive_inst<detection_output> : public typed_primitive_inst_base<detection_output> {
    using parent = typed_primitive_inst_base<detection_output>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(detection_output_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(detection_output_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(detection_output_node const& node);

    typed_primitive_inst(network& network, detection_output_node const& node);

    memory::ptr location_memory() const { return dep_memory_ptr(0); }
    memory::ptr confidence_memory() const { return dep_memory_ptr(1); }
    memory::ptr prior_box_memory() const { return dep_memory_ptr(2); }
};

using detection_output_inst = typed_primitive_inst<detection_output>;

}  // namespace cldnn
