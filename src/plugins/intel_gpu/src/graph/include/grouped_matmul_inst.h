// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/grouped_matmul.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<grouped_matmul> : public typed_program_node_base<grouped_matmul> {
    using parent = typed_program_node_base<grouped_matmul>;
    typed_program_node(const std::shared_ptr<grouped_matmul> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
    program_node& input() const { return get_dependency(0); }
};

using grouped_matmul_node = typed_program_node<grouped_matmul>;

template <>
class typed_primitive_inst<grouped_matmul> : public typed_primitive_inst_base<grouped_matmul> {
    using parent = typed_primitive_inst_base<grouped_matmul>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(grouped_matmul_node const& node, const kernel_impl_params& impl_param);
    static layout calc_output_layout(grouped_matmul_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(grouped_matmul_node const& node);

    typed_primitive_inst(network& network, grouped_matmul_node const& node);
};

using grouped_matmul_inst = typed_primitive_inst<grouped_matmul>;
}  // namespace cldnn
