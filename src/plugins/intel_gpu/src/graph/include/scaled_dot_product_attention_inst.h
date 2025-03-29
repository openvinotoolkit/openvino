// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<scaled_dot_product_attention> : public typed_program_node_base<scaled_dot_product_attention> {
    using parent = typed_program_node_base<scaled_dot_product_attention>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};
using scaled_dot_product_attention_node = typed_program_node<scaled_dot_product_attention>;

template <>
class typed_primitive_inst<scaled_dot_product_attention> : public typed_primitive_inst_base<scaled_dot_product_attention> {
    using parent = typed_primitive_inst_base<scaled_dot_product_attention>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(scaled_dot_product_attention_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(scaled_dot_product_attention_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(scaled_dot_product_attention_node const& node);
    bool has_indirect_inputs() const {
        return get_typed_desc<scaled_dot_product_attention>()->indirect_axis != -1;
    }

    typed_primitive_inst(network& network, scaled_dot_product_attention_node const& desc);
};

using scaled_dot_product_attention_inst = typed_primitive_inst<scaled_dot_product_attention>;
}  // namespace cldnn
