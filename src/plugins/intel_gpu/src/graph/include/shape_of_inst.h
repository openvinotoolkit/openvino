// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/shape_of.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<shape_of> : public typed_program_node_base<shape_of> {
    using parent = typed_program_node_base<shape_of>;
    typed_program_node(const std::shared_ptr<shape_of> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
    program_node& input() const { return get_dependency(0); }
};

using shape_of_node = typed_program_node<shape_of>;

template <>
class typed_primitive_inst<shape_of> : public typed_primitive_inst_base<shape_of> {
    using parent = typed_primitive_inst_base<shape_of>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(shape_of_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(shape_of_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(shape_of_node const& node);

    typed_primitive_inst(network& network, shape_of_node const& node);
};

using shape_of_inst = typed_primitive_inst<shape_of>;
}  // namespace cldnn
