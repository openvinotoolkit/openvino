// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/batch_to_space.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<batch_to_space> : public typed_program_node_base<batch_to_space> {
    using parent = typed_program_node_base<batch_to_space>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {1, 2, 3}; }
};
using batch_to_space_node = typed_program_node<batch_to_space>;

template <>
class typed_primitive_inst<batch_to_space> : public typed_primitive_inst_base<batch_to_space> {
    using parent = typed_primitive_inst_base<batch_to_space>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(batch_to_space_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(batch_to_space_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(batch_to_space_node const& node);

    bool need_reset_output_memory() const override {
        const auto desc = _impl_params->typed_desc<batch_to_space>();
        if (desc->shape_constant) {
            return true;
        }

        return false;
    }

    typed_primitive_inst(network& network, batch_to_space_node const& desc);
};

using batch_to_space_inst = typed_primitive_inst<batch_to_space>;
}  // namespace cldnn
