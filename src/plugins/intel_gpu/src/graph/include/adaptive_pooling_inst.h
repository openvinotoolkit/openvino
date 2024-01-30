// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/adaptive_pooling.hpp"
#include "primitive_inst.h"
#include <string>

namespace cldnn {
template <>
struct typed_program_node<adaptive_pooling> : public typed_program_node_base<adaptive_pooling> {
    using parent = typed_program_node_base<adaptive_pooling>;

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {1}; }
};

using adaptive_pooling_node = typed_program_node<adaptive_pooling>;

template <>
class typed_primitive_inst<adaptive_pooling> : public typed_primitive_inst_base<adaptive_pooling> {
    using parent = typed_primitive_inst_base<adaptive_pooling>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(adaptive_pooling_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(adaptive_pooling_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(adaptive_pooling_node const& node);

    typed_primitive_inst(network& network, adaptive_pooling_node const& node);
};

using adaptive_pooling_inst = typed_primitive_inst<adaptive_pooling>;
}  // namespace cldnn
