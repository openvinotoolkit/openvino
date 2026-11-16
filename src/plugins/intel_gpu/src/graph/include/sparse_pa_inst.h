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

    // Output shapes come from the query alone; past_lens / subsequence_begins stay in GPU
    // memory (the impl sizes everything from shape-derived bounds), so no host readback.
    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {};
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
