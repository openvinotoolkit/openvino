// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/cum_sum.hpp"
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<cum_sum> : public typed_program_node_base<cum_sum> {
    using parent = typed_program_node_base<cum_sum>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using cum_sum_node = typed_program_node<cum_sum>;

template <>
class typed_primitive_inst<cum_sum> : public typed_primitive_inst_base<cum_sum> {
    using parent = typed_primitive_inst_base<cum_sum>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(cum_sum_node const& /*node*/, const kernel_impl_params& impl_param) {
        return forward_input0_shape<ShapeType>(impl_param);
    }

    static layout calc_output_layout(cum_sum_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(cum_sum_node const& node);
    typed_primitive_inst(network& network, cum_sum_node const& desc);
};

using cum_sum_inst = typed_primitive_inst<cum_sum>;
}  // namespace cldnn
