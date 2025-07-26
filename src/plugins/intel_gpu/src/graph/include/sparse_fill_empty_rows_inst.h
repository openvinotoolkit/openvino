// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <intel_gpu/primitives/sparse_fill_empty_rows.hpp>
#include "primitive_inst.h"

namespace cldnn {
template <>
struct typed_program_node<sparse_fill_empty_rows> : public typed_program_node_base<sparse_fill_empty_rows> {
    using parent = typed_program_node_base<sparse_fill_empty_rows>;
    typed_program_node(const std::shared_ptr<sparse_fill_empty_rows> prim, program& prog) : parent(prim, prog) {}

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const {
        return get_dependency(idx);
    }
    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {1, 2};
    }
};

using SparseFillEmptyRows_node = typed_program_node<sparse_fill_empty_rows>;

template <>
class typed_primitive_inst<sparse_fill_empty_rows> : public typed_primitive_inst_base<sparse_fill_empty_rows> {
    using parent = typed_primitive_inst_base<sparse_fill_empty_rows>;
    using parent::parent;

public:
    typed_primitive_inst(network& network, SparseFillEmptyRows_node const& desc);
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(SparseFillEmptyRows_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(SparseFillEmptyRows_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(SparseFillEmptyRows_node const& node);
    bool need_reset_output_memory() const override {
        return true;
    }
};

using SparseFillEmptyRows_inst = typed_primitive_inst<sparse_fill_empty_rows>;
}  // namespace cldnn
