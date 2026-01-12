// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/moe_scatter_reduction.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<moe_scatter_reduction> : public typed_program_node_base<moe_scatter_reduction> {
    using parent = typed_program_node_base<moe_scatter_reduction>;
    typed_program_node(const std::shared_ptr<moe_scatter_reduction> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
    program_node& input() const { return get_dependency(0); }
};

using moe_scatter_reduction_node = typed_program_node<moe_scatter_reduction>;

template <>
class typed_primitive_inst<moe_scatter_reduction> : public typed_primitive_inst_base<moe_scatter_reduction> {
    using parent = typed_primitive_inst_base<moe_scatter_reduction>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(moe_scatter_reduction_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(moe_scatter_reduction_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(moe_scatter_reduction_node const& node);

    typed_primitive_inst(network& network, moe_scatter_reduction_node const& node);
};

using moe_scatter_reduction_inst = typed_primitive_inst<moe_scatter_reduction>;
}  // namespace cldnn

