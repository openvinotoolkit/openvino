// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/concatenation.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<concatenation> : public typed_program_node_base<concatenation> {
    using parent = typed_program_node_base<concatenation>;
    typed_program_node(const std::shared_ptr<concatenation> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using concatenation_node = typed_program_node<concatenation>;

template <>
class typed_primitive_inst<concatenation> : public typed_primitive_inst_base<concatenation> {
    using parent = typed_primitive_inst_base<concatenation>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(const concatenation_node& /* node */, const kernel_impl_params& impl_param);
    static layout calc_output_layout(concatenation_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(concatenation_node const& node);

    typed_primitive_inst(network& network, concatenation_node const& node);
};

using concatenation_inst = typed_primitive_inst<concatenation>;

}  // namespace cldnn
