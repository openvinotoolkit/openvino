// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/col2im.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {
template <>
struct typed_program_node<col2im> : public typed_program_node_base<col2im> {
    using parent = typed_program_node_base<col2im>;

public:
    using parent::parent;

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
};

using col2im_node = typed_program_node<col2im>;

template <>
class typed_primitive_inst<col2im> : public typed_primitive_inst_base<col2im> {
    using parent = typed_primitive_inst_base<col2im>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(const col2im_node& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(col2im_node const& node, kernel_impl_params const& impl_param);

    static bool validate_num_blocks(kernel_impl_params const& impl_param, size_t candidate_num_blocks);

    static std::string to_string(col2im_node const& node);
};

using col2im_inst = typed_primitive_inst<col2im>;
}  // namespace cldnn
