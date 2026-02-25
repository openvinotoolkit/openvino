// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <intel_gpu/primitives/eye.hpp>

#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<eye> : typed_program_node_base<eye> {
private:
    using parent = typed_program_node_base<eye>;

public:
    using parent::parent;

    typed_program_node(const std::shared_ptr<eye> prim, program& prog) : parent(prim, prog) { }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {0, 1, 2, 3}; }
};
using eye_node = typed_program_node<eye>;

template <>
class typed_primitive_inst<eye> : public typed_primitive_inst_base<eye> {
    using parent = typed_primitive_inst_base<eye>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(eye_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(eye_node const& node, const kernel_impl_params& impl_param);
    static std::string to_string(eye_node const& node);

    typed_primitive_inst(network& network, eye_node const& desc);
};

using eye_inst = typed_primitive_inst<eye>;

}  // namespace cldnn
