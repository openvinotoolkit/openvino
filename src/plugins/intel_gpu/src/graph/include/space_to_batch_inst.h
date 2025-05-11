// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/space_to_batch.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<space_to_batch> : public typed_program_node_base<space_to_batch> {
    using parent = typed_program_node_base<space_to_batch>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {1, 2, 3}; }
};

using space_to_batch_node = typed_program_node<space_to_batch>;

template <>
class typed_primitive_inst<space_to_batch> : public typed_primitive_inst_base<space_to_batch> {
    using parent = typed_primitive_inst_base<space_to_batch>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(space_to_batch_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(space_to_batch_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(space_to_batch_node const& node);

    bool need_reset_output_memory() const override {
        const auto desc = _impl_params->typed_desc<space_to_batch>();
        if (!desc->shape_constant) {
            return true;
        }

        return false;
    }

    typed_primitive_inst(network& network, space_to_batch_node const& desc);
};

using space_to_batch_inst = typed_primitive_inst<space_to_batch>;
}  // namespace cldnn
