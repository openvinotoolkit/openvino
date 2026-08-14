// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/linear_attention.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<linear_attention> : public typed_program_node_base<linear_attention> {
    using parent = typed_program_node_base<linear_attention>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};
using linear_attention_node = typed_program_node<linear_attention>;

template <>
class typed_primitive_inst<linear_attention> : public typed_primitive_inst_base<linear_attention> {
    using parent = typed_primitive_inst_base<linear_attention>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(const linear_attention_node& /*node*/, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const linear_attention_node& node, const kernel_impl_params& impl_params);

    static std::string to_string(const linear_attention_node& node);

    typed_primitive_inst(network& network, const linear_attention_node& node);

};

using linear_attention_inst = typed_primitive_inst<linear_attention>;
}  // namespace cldnn
