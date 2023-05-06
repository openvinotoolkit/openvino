// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/reshape.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<reshape> : public typed_program_node_base<reshape> {
    using parent = typed_program_node_base<reshape>;
    typed_program_node(const std::shared_ptr<reshape> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }

    bool is_in_place() const {
        if (this->is_output() || this->has_fused_primitives())
            return false;
        return (!this->get_output_layout().data_padding && !input().get_output_layout(false).data_padding);
    }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {1}; }
};

using reshape_node = typed_program_node<reshape>;

template <>
class typed_primitive_inst<reshape> : public typed_primitive_inst_base<reshape> {
    using parent = typed_primitive_inst_base<reshape>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(reshape_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(reshape_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(reshape_node const& node);

    typed_primitive_inst(network& network, reshape_node const& node);

    void update_output_memory() override;

private:
    void on_execute() override;

    void reuse_input();
};

using reshape_inst = typed_primitive_inst<reshape>;

}  // namespace cldnn
