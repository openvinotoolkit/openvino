// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/input_layout.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<input_layout> : public typed_program_node_base<input_layout> {
    using parent = typed_program_node_base<input_layout>;
    using parent::parent;

    typed_program_node(const std::shared_ptr<input_layout> prim, program& prog);
};

using input_layout_node = typed_program_node<input_layout>;

template <>
class typed_primitive_inst<input_layout> : public typed_primitive_inst_base<input_layout> {
    using parent = typed_primitive_inst_base<input_layout>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(input_layout_node const& /* node */, const kernel_impl_params& impl_param) {
        return { impl_param.typed_desc<input_layout>()->layout };
    }

    static layout calc_output_layout(input_layout_node const& node, kernel_impl_params const& impl_param) {
        return impl_param.typed_desc<input_layout>()->layout;
    }
    static std::string to_string(input_layout_node const& node);

    void update_shape() override;
    typed_primitive_inst(network& network, input_layout_node const& node);

    event::ptr set_data(memory::ptr mem);
};

using input_layout_inst = typed_primitive_inst<input_layout>;

}  // namespace cldnn
