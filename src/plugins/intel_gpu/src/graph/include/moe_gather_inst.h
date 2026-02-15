// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/moe_gather.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<moe_gather> : public typed_program_node_base<moe_gather> {
    using parent = typed_program_node_base<moe_gather>;
    typed_program_node(const std::shared_ptr<moe_gather> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
    program_node& input() const { return get_dependency(0); }
};

using moe_gather_node = typed_program_node<moe_gather>;

template <>
class typed_primitive_inst<moe_gather> : public typed_primitive_inst_base<moe_gather> {
    using parent = typed_primitive_inst_base<moe_gather>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(moe_gather_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(moe_gather_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(moe_gather_node const& node);

    typed_primitive_inst(network& network, moe_gather_node const& node);
};

using moe_gather_inst = typed_primitive_inst<moe_gather>;
}  // namespace cldnn
