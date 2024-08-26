// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/fc_shape_of.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<fc_shape_of> : public typed_program_node_base<fc_shape_of> {
    using parent = typed_program_node_base<fc_shape_of>;
    typed_program_node(const std::shared_ptr<fc_shape_of> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
    program_node& input() const { return get_dependency(0); }
};

using fc_shape_of_node = typed_program_node<fc_shape_of>;

template <>
class typed_primitive_inst<fc_shape_of> : public typed_primitive_inst_base<fc_shape_of> {
    using parent = typed_primitive_inst_base<fc_shape_of>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(fc_shape_of_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(fc_shape_of_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(fc_shape_of_node const& node);

    typed_primitive_inst(network& network, fc_shape_of_node const& node);
};

using fc_shape_of_inst = typed_primitive_inst<fc_shape_of>;
}  // namespace cldnn
