// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/grid_sample.hpp"
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<grid_sample> : public typed_program_node_base<grid_sample> {
    using parent = typed_program_node_base<grid_sample>;

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using grid_sample_node = typed_program_node<grid_sample>;

template <>
class typed_primitive_inst<grid_sample> : public typed_primitive_inst_base<grid_sample> {
    using parent = typed_primitive_inst_base<grid_sample>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(grid_sample_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const grid_sample_node& node, const kernel_impl_params& impl_param);
    static std::string to_string(const grid_sample_node& node);

    typed_primitive_inst(network& network, grid_sample_node const& node);
};

using grid_sample_inst = typed_primitive_inst<grid_sample>;

}  // namespace cldnn
