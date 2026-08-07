// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "intel_gpu/primitives/mlir_primitive.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<mlir_primitive> : public typed_program_node_base<mlir_primitive> {
    using parent = typed_program_node_base<mlir_primitive>;

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
};

using mlir_primitive_node = typed_program_node<mlir_primitive>;

template <>
class typed_primitive_inst<mlir_primitive> : public typed_primitive_inst_base<mlir_primitive> {
    using parent = typed_primitive_inst_base<mlir_primitive>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(mlir_primitive_node const& node,
                                                   const kernel_impl_params& impl_param);
    static layout calc_output_layout(mlir_primitive_node const& node, kernel_impl_params const& impl_param);

    static std::string to_string(mlir_primitive_node const& node);

    typed_primitive_inst(network& network, mlir_primitive_node const& node);

    const mlir_primitive_node* node;
};

using mlir_primitive_inst = typed_primitive_inst<mlir_primitive>;

}  // namespace cldnn
