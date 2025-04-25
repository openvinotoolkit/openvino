// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/one_hot.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<one_hot> : typed_program_node_base<one_hot> {
private:
    using parent = typed_program_node_base<one_hot>;

public:
    using parent::parent;

    typed_program_node(const std::shared_ptr<one_hot> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }
    program_node& input() const { return get_dependency(0); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {1}; }
};

using one_hot_node = typed_program_node<one_hot>;

template <>
class typed_primitive_inst<one_hot> : public typed_primitive_inst_base<one_hot> {
    using parent = typed_primitive_inst_base<one_hot>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(const one_hot_node& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(one_hot_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(one_hot_node const& node);
    typed_primitive_inst(network& network, one_hot_node const& node);
};

using one_hot_inst = typed_primitive_inst<one_hot>;
}  // namespace cldnn
