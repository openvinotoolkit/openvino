// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "intel_gpu/primitives/gated_residual.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<gated_residual> : public typed_program_node_base<gated_residual> {
    using parent = typed_program_node_base<gated_residual>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const {
        return get_dependency(index);
    }
    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {};
    }
};
using gated_residual_node = typed_program_node<gated_residual>;

template <>
class typed_primitive_inst<gated_residual> : public typed_primitive_inst_base<gated_residual> {
    using parent = typed_primitive_inst_base<gated_residual>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const gated_residual_node& node, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const gated_residual_node& node, const kernel_impl_params& impl_params);

    static std::string to_string(const gated_residual_node& node);
    typed_primitive_inst(network& network, const gated_residual_node& node);
};

using gated_residual_inst = typed_primitive_inst<gated_residual>;

}  // namespace cldnn
