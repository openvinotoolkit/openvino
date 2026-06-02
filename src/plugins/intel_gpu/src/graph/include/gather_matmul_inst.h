// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>
#include <string>

#include "intel_gpu/primitives/gather_matmul.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<gather_matmul> : public typed_program_node_base<gather_matmul> {
    using parent = typed_program_node_base<gather_matmul>;
    typed_program_node(const std::shared_ptr<gather_matmul> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {};
    }
    program_node& input() const {
        return get_dependency(0);
    }
};

using gather_matmul_node = typed_program_node<gather_matmul>;

template <>
class typed_primitive_inst<gather_matmul> : public typed_primitive_inst_base<gather_matmul> {
    using parent = typed_primitive_inst_base<gather_matmul>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const gather_matmul_node& node, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const gather_matmul_node& node, const kernel_impl_params& impl_param);
    static std::string to_string(const gather_matmul_node& node);

    typed_primitive_inst(network& network, const gather_matmul_node& node);
};

using gather_matmul_inst = typed_primitive_inst<gather_matmul>;
}  // namespace cldnn
