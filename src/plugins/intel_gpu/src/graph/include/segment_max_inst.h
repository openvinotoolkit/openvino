// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <intel_gpu/primitives/segment_max.hpp>

#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<segment_max> : public typed_program_node_base<segment_max> {
    using parent = typed_program_node_base<segment_max>;
    typed_program_node(const std::shared_ptr<segment_max> prim, program& prog) : parent(prim, prog) {}

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const {
        return get_dependency(idx);
    }
    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {};
    }
};

using segment_max_node = typed_program_node<segment_max>;

template <>
class typed_primitive_inst<segment_max> : public typed_primitive_inst_base<segment_max> {
    using parent = typed_primitive_inst_base<segment_max>;
    using parent::parent;

public:
    typed_primitive_inst(network& network, segment_max_node const& desc);
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(segment_max_node const& node,
                                                   kernel_impl_params const& impl_param);
    static layout calc_output_layout(segment_max_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(segment_max_node const& node);
};

using segment_max_inst = typed_primitive_inst<segment_max>;

}  // namespace cldnn
