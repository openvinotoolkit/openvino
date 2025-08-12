// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/vl_sdpa.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<vl_sdpa> : public typed_program_node_base<vl_sdpa> {
    using parent = typed_program_node_base<vl_sdpa>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};
using vl_sdpa_node = typed_program_node<vl_sdpa>;

template <>
class typed_primitive_inst<vl_sdpa> : public typed_primitive_inst_base<vl_sdpa> {
    using parent = typed_primitive_inst_base<vl_sdpa>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(const vl_sdpa_node& /*node*/, const kernel_impl_params& impl_params) {
        return forward_input0_shape<ShapeType>(impl_params);
    }
    static layout calc_output_layout(const vl_sdpa_node& node, const kernel_impl_params& impl_params) {
        return calc_output_layouts<ov::PartialShape>(node, impl_params)[0];
    }

    static std::string to_string(const vl_sdpa_node& node);

    typed_primitive_inst(network& network, const vl_sdpa_node& node);

    void get_mask_seqlens_from_memory(std::vector<int32_t>& cu_seqlens) const;

    memory::ptr cu_seqlens_memory_ptr() const { return dep_memory_ptr(3); }
};

using vl_sdpa_inst = typed_primitive_inst<vl_sdpa>;
}  // namespace cldnn
