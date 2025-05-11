// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/softmax.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {
template <>
struct typed_program_node<softmax> : public typed_program_node_base<softmax> {
    using parent = typed_program_node_base<softmax>;

public:
    using parent::parent;

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};
using softmax_node = typed_program_node<softmax>;

template <>
class typed_primitive_inst<softmax> : public typed_primitive_inst_base<softmax> {
    using parent = typed_primitive_inst_base<softmax>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(softmax_node const& /*node*/, const kernel_impl_params& impl_param) {
        return forward_input0_shape<ShapeType>(impl_param);
    }

    static layout calc_output_layout(softmax_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(softmax_node const& node);

    typed_primitive_inst(network& network, softmax_node const& desc);
};

using softmax_inst = typed_primitive_inst<softmax>;

}  // namespace cldnn
