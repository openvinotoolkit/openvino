// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/moe.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>
#include <vector>

namespace cldnn {
namespace details {}

template <>
struct typed_program_node<moe> : public typed_program_node_base<moe> {
private:
    using parent = typed_program_node_base<moe>;

public:
    using parent::parent;

    typed_program_node(std::shared_ptr<moe> prim, program& prog) : parent(prim, prog) {}

    using parent::get_kernel_impl_params;
    std::unique_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts, const std::vector<layout>& out_layouts) const override {
        auto params = parent::get_kernel_impl_params(in_layouts, out_layouts);

        return params;
    }
};

using moe_node = typed_program_node<moe>;

template <>
class typed_primitive_inst<moe> : public typed_primitive_inst_base<moe> {
    using parent = typed_primitive_inst_base<moe>;
    using parent::parent;
    using primitive_inst::update_output_memory;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(moe_node const& /*node*/, kernel_impl_params const& impl_param);
    static layout calc_output_layout(moe_node const& /* node */, kernel_impl_params const& impl_param);
    static std::string to_string(moe_node const& node);
    typed_primitive_inst(network& network, moe_node const& node);
};

using moe_inst = typed_primitive_inst<moe>;
}  // namespace cldnn
