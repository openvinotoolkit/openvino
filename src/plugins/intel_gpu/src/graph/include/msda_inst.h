// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/msda.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

template <>
struct typed_program_node<msda> : public typed_program_node_base<msda> {
    using parent = typed_program_node_base<msda>;

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};
using msda_node = typed_program_node<msda>;

template <>
class typed_primitive_inst<msda> : public typed_primitive_inst_base<msda> {
    using parent = typed_primitive_inst_base<msda>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(const msda_node& /*node*/, const kernel_impl_params& impl_params) {
        return forward_input0_shape<ShapeType>(impl_params);
    }
    static layout calc_output_layout(const msda_node& node, const kernel_impl_params& impl_params) {
        return calc_output_layouts<ov::PartialShape>(node, impl_params)[0];
    }

    static std::string to_string(const msda_node& node);

    typed_primitive_inst(network& network, const msda_node& node);

    std::vector<int32_t> get_mask_seqlens_from_memory() const;
    static std::vector<int32_t> get_mask_seqlens_from_memory2(memory::ptr mem, stream& stream);
};

using msda_inst = typed_primitive_inst<msda>;
}  // namespace cldnn