// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/border.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<border> : typed_program_node_base<border> {
private:
    using parent = typed_program_node_base<border>;

public:
    using parent::parent;

    typed_program_node(const std::shared_ptr<border> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }
    program_node& input() const { return get_dependency(0); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {1, 2, 3}; }
};

using border_node = typed_program_node<border>;

template <>
class typed_primitive_inst<border> : public typed_primitive_inst_base<border> {
    using parent = typed_primitive_inst_base<border>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(border_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(border_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(border_node const& node);
    typed_primitive_inst(network& network, border_node const& node);
};

using border_inst = typed_primitive_inst<border>;
}  // namespace cldnn
