// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/gated_mlp.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<gated_mlp> : public typed_program_node_base<gated_mlp> {
    using parent = typed_program_node_base<gated_mlp>;
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    program_node& weights_gate() const { return get_dependency(1); }
    program_node& weights_up() const { return get_dependency(2); }
    program_node& weights_down() const { return get_dependency(3); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using gated_mlp_node = typed_program_node<gated_mlp>;

template <>
class typed_primitive_inst<gated_mlp> : public typed_primitive_inst_base<gated_mlp> {
    using parent = typed_primitive_inst_base<gated_mlp>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(gated_mlp_node const& node, const kernel_impl_params& impl_param);
    static layout calc_output_layout(gated_mlp_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(gated_mlp_node const& node);

    typed_primitive_inst(network& network, gated_mlp_node const& node);
};

using gated_mlp_inst = typed_primitive_inst<gated_mlp>;

}  // namespace cldnn
