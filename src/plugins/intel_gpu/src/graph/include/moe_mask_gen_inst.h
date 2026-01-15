// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/moe_mask_gen.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<moe_mask_gen> : public typed_program_node_base<moe_mask_gen> {
    using parent = typed_program_node_base<moe_mask_gen>;
    typed_program_node(const std::shared_ptr<moe_mask_gen> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
    program_node& input() const { return get_dependency(0); }
};

using moe_mask_gen_node = typed_program_node<moe_mask_gen>;

template <>
class typed_primitive_inst<moe_mask_gen> : public typed_primitive_inst_base<moe_mask_gen> {
    using parent = typed_primitive_inst_base<moe_mask_gen>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(moe_mask_gen_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(moe_mask_gen_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(moe_mask_gen_node const& node);

    typed_primitive_inst(network& network, moe_mask_gen_node const& node);
};

using moe_mask_gen_inst = typed_primitive_inst<moe_mask_gen>;


template <>
struct typed_program_node<moe_mask_gen_reshape> : public typed_program_node_base<moe_mask_gen_reshape> {
    using parent = typed_program_node_base<moe_mask_gen_reshape>;
    typed_program_node(const std::shared_ptr<moe_mask_gen_reshape> prim, program& prog) : parent(prim, prog) {
        can_be_optimized(true);
    }

public:
    using parent::parent;

    std::vector<size_t> get_shape_infer_dependencies() const override { return {4}; }
    program_node& input() const { return get_dependency(0); }
};

using moe_mask_gen_reshape_node = typed_program_node<moe_mask_gen_reshape>;

template <>
class typed_primitive_inst<moe_mask_gen_reshape> : public typed_primitive_inst_base<moe_mask_gen_reshape> {
    using parent = typed_primitive_inst_base<moe_mask_gen_reshape>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(moe_mask_gen_reshape_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(moe_mask_gen_reshape_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(moe_mask_gen_reshape_node const& node);

    typed_primitive_inst(network& network, moe_mask_gen_reshape_node const& node);
    void update_output_memory() override;
private:
    void on_execute() override;
};

using moe_mask_gen_reshape_inst = typed_primitive_inst<moe_mask_gen_reshape>;

}  // namespace cldnn
