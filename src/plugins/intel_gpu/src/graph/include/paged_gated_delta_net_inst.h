// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/paged_gated_delta_net.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<paged_gated_delta_net> : public typed_program_node_base<paged_gated_delta_net> {
    using parent = typed_program_node_base<paged_gated_delta_net>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};
using paged_gated_delta_net_node = typed_program_node<paged_gated_delta_net>;

template <>
class typed_primitive_inst<paged_gated_delta_net> : public typed_primitive_inst_base<paged_gated_delta_net> {
    using parent = typed_primitive_inst_base<paged_gated_delta_net>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const paged_gated_delta_net_node& node, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const paged_gated_delta_net_node& node, const kernel_impl_params& impl_params);

    static std::string to_string(const paged_gated_delta_net_node& node);
    typed_primitive_inst(network& network, const paged_gated_delta_net_node& node);
};

using paged_gated_delta_net_inst = typed_primitive_inst<paged_gated_delta_net>;
}  // namespace cldnn
