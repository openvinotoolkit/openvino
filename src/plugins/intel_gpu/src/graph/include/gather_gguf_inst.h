// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/gather_gguf.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

template <>
struct typed_program_node<gather_gguf> : public typed_program_node_base<gather_gguf> {
    using parent = typed_program_node_base<gather_gguf>;
    typed_program_node(const std::shared_ptr<gather_gguf> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    // Indices (input 1) shape drives the output shape; keep it as a shape-infer dependency.
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
    program_node& input() const { return get_dependency(0); }
    program_node& indices() const { return get_dependency(1); }
};

using gather_gguf_node = typed_program_node<gather_gguf>;

template <>
class typed_primitive_inst<gather_gguf> : public typed_primitive_inst_base<gather_gguf> {
    using parent = typed_primitive_inst_base<gather_gguf>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(gather_gguf_node const& node, const kernel_impl_params& impl_param);
    static layout calc_output_layout(gather_gguf_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(gather_gguf_node const& node);

    typed_primitive_inst(network& network, gather_gguf_node const& node);
};

using gather_gguf_inst = typed_primitive_inst<gather_gguf>;
}  // namespace cldnn
