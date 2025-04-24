// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>
#include <string>

#include "intel_gpu/primitives/matrix_nms.hpp"
#include "primitive_inst.h"

namespace cldnn {

using matrix_nms_node = typed_program_node<matrix_nms>;

template <>
class typed_primitive_inst<matrix_nms> : public typed_primitive_inst_base<matrix_nms> {
    using parent = typed_primitive_inst_base<matrix_nms>;
    using parent::parent;

public:
    typed_primitive_inst(network& network, const matrix_nms_node& node) : parent(network, node) {}

    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(matrix_nms_node const& /*node*/, const kernel_impl_params& impl_param);

    static layout calc_output_layout(const matrix_nms_node& node, const kernel_impl_params& impl_param);
    static std::string to_string(const matrix_nms_node& node);

    memory::ptr input_boxes_mem() const {
        return dep_memory_ptr(0);
    }
    memory::ptr input_scores_mem() const {
        return dep_memory_ptr(1);
    }
    memory::ptr input_selected_boxes_mem() const {
        return dep_memory_ptr(2);
    }
    memory::ptr input_valid_outputs_mem() const {
        return dep_memory_ptr(3);
    }
};

using matrix_nms_inst = typed_primitive_inst<matrix_nms>;

}  // namespace cldnn
