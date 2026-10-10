// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <intel_gpu/primitives/bevpool_v2.hpp>

#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<bevpool_v2> : public typed_program_node_base<bevpool_v2> {
    using parent = typed_program_node_base<bevpool_v2>;
    typed_program_node(const std::shared_ptr<bevpool_v2> prim, program& prog) : parent(prim, prog) {}

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const {
        return get_dependency(idx);
    }

    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {};
    }
};

using bevpool_v2_node = typed_program_node<bevpool_v2>;

template <>
class typed_primitive_inst<bevpool_v2> : public typed_primitive_inst_base<bevpool_v2> {
    using parent = typed_primitive_inst_base<bevpool_v2>;
    using parent::parent;

public:
    typed_primitive_inst(network& network, const bevpool_v2_node& desc);
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const bevpool_v2_node& node, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const bevpool_v2_node& node, const kernel_impl_params& impl_param);
    static std::string to_string(const bevpool_v2_node& node);
};

using bevpool_v2_inst = typed_primitive_inst<bevpool_v2>;

}  // namespace cldnn
