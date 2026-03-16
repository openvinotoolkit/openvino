// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/gated_delta_net.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<gated_delta_net> : public typed_program_node_base<gated_delta_net> {
    using parent = typed_program_node_base<gated_delta_net>;
public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};
using gated_delta_net_node = typed_program_node<gated_delta_net>;

template <>
class typed_primitive_inst<gated_delta_net> : public typed_primitive_inst_base<gated_delta_net> {
    using parent = typed_primitive_inst_base<gated_delta_net>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(const gated_delta_net_node& /*node*/, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const gated_delta_net_node& node, const kernel_impl_params& impl_params);

    static std::string to_string(const gated_delta_net_node& node);
    typed_primitive_inst(network& network, const gated_delta_net_node& node);

};

using gated_delta_net_inst = typed_primitive_inst<gated_delta_net>;
}  // namespace cldnn
