// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <intel_gpu/primitives/search_sorted.hpp>

#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<search_sorted> : public typed_program_node_base<search_sorted> {
    using parent = typed_program_node_base<search_sorted>;
    typed_program_node(const std::shared_ptr<search_sorted> prim, program& prog) : parent(prim, prog) {}

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const {
        return get_dependency(idx);
    }
    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {};
    }
};

using search_sorted_node = typed_program_node<search_sorted>;

template <>
class typed_primitive_inst<search_sorted> : public typed_primitive_inst_base<search_sorted> {
    using parent = typed_primitive_inst_base<search_sorted>;
    using parent::parent;

public:
    typed_primitive_inst(network& network, search_sorted_node const& desc);
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(search_sorted_node const& node,
                                                   kernel_impl_params const& impl_param);
    static layout calc_output_layout(search_sorted_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(search_sorted_node const& node);
};

using search_sorted_inst = typed_primitive_inst<search_sorted>;

}  // namespace cldnn
