// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "intel_gpu/primitives/sparse_pa.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<sparse_pa> : public typed_program_node_base<sparse_pa> {
    using parent = typed_program_node_base<sparse_pa>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const {
        return get_dependency(index);
    }

    // past_lens / subsequence_begins must be host-readable so the CM implementation can
    // size its internal scratch and build its dispatch maps.
    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {ov::intel_gpu::op::SparsePA::InputIdx::PAST_LENS, ov::intel_gpu::op::SparsePA::InputIdx::SUBSEQUENCE_BEGINS};
    }
};
using sparse_pa_node = typed_program_node<sparse_pa>;

template <>
class typed_primitive_inst<sparse_pa> : public typed_primitive_inst_base<sparse_pa> {
    using parent = typed_primitive_inst_base<sparse_pa>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const sparse_pa_node& node, const kernel_impl_params& impl_params);
    static layout calc_output_layout(const sparse_pa_node& node, const kernel_impl_params& impl_params);

    static std::string to_string(const sparse_pa_node& node);
    typed_primitive_inst(network& network, const sparse_pa_node& node);
};

using sparse_pa_inst = typed_primitive_inst<sparse_pa>;

}  // namespace cldnn
