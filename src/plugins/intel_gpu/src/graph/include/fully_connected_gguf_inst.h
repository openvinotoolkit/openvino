// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <memory>
#include <string>

#include "intel_gpu/primitives/fully_connected_gguf.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<fully_connected_gguf> : public typed_program_node_base<fully_connected_gguf> {
    using parent = typed_program_node_base<fully_connected_gguf>;
    typed_program_node(const std::shared_ptr<fully_connected_gguf> prim, program& prog) : parent(prim, prog) {}

public:
    using parent::parent;

    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {};
    }
};

using fully_connected_gguf_node = typed_program_node<fully_connected_gguf>;

template <>
class typed_primitive_inst<fully_connected_gguf> : public typed_primitive_inst_base<fully_connected_gguf> {
    using parent = typed_primitive_inst_base<fully_connected_gguf>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const fully_connected_gguf_node& node, const kernel_impl_params& impl_param);
    static layout calc_output_layout(const fully_connected_gguf_node& node, const kernel_impl_params& impl_param);
    static std::string to_string(const fully_connected_gguf_node& node);

    typed_primitive_inst(network& network, const fully_connected_gguf_node& node);
};

using fully_connected_gguf_inst = typed_primitive_inst<fully_connected_gguf>;
}  // namespace cldnn
