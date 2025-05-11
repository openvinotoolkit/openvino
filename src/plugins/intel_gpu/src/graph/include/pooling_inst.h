// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/pooling.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<pooling> : public typed_program_node_base<pooling> {
    using parent = typed_program_node_base<pooling>;
    typed_program_node(const std::shared_ptr<pooling> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;
    program_node& input() const { return get_dependency(0); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using pooling_node = typed_program_node<pooling>;

template <>
class typed_primitive_inst<pooling> : public typed_primitive_inst_base<pooling> {
    using parent = typed_primitive_inst_base<pooling>;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(pooling_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(pooling_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(pooling_node const& node);

public:
    using parent::parent;
};

using pooling_inst = typed_primitive_inst<pooling>;

}  // namespace cldnn
