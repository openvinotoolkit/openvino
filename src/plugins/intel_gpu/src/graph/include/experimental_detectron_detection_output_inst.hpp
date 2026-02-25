// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/experimental_detectron_detection_output.hpp"
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<experimental_detectron_detection_output>
    : public typed_program_node_base<experimental_detectron_detection_output> {
    using parent = typed_program_node_base<experimental_detectron_detection_output>;

public:
    using parent::parent;

    program_node& input() const {
        return get_dependency(0);
    }

    program_node& deltas() const {
        return get_dependency(1);
    }
    program_node& scores() const {
        return get_dependency(2);
    }
    program_node& image_size_info() const {
        return get_dependency(3);
    }

    program_node& output_classes_node() const {
        return get_dependency(4);
    }
    program_node& output_scores_node() const {
        return get_dependency(5);
    }
};

using experimental_detectron_detection_output_node = typed_program_node<experimental_detectron_detection_output>;

template <>
class typed_primitive_inst<experimental_detectron_detection_output>
    : public typed_primitive_inst_base<experimental_detectron_detection_output> {
    using parent = typed_primitive_inst_base<experimental_detectron_detection_output>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(experimental_detectron_detection_output_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const experimental_detectron_detection_output_node& node, kernel_impl_params const& impl_param);
    static std::string to_string(const experimental_detectron_detection_output_node& node);

    typed_primitive_inst(network& network, const experimental_detectron_detection_output_node& node)
        : parent(network, node) {}

    memory::ptr output_classes_memory() const {
        return dep_memory_ptr(4);
    }
    memory::ptr output_scores_memory() const {
        return dep_memory_ptr(5);
    }
};

using experimental_detectron_detection_output_inst = typed_primitive_inst<experimental_detectron_detection_output>;

}  // namespace cldnn
