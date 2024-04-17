// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/swiglu.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<swiglu> : public typed_program_node_base<swiglu> {
    using parent = typed_program_node_base<swiglu>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using swiglu_node = typed_program_node<swiglu>;

template <>
class typed_primitive_inst<swiglu> : public typed_primitive_inst_base<swiglu> {
    using parent = typed_primitive_inst_base<swiglu>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(swiglu_node const& /*node*/, const kernel_impl_params& impl_params);
    static std::string to_string(swiglu_node const& node);

    typed_primitive_inst(network& network, swiglu_node const& node);
};

using swiglu_inst = typed_primitive_inst<swiglu>;

}  // namespace cldnn
