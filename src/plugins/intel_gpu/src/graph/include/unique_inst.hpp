// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/unique.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<unique> : typed_program_node_base<unique> {
    using parent = typed_program_node_base<unique>;
    using parent::parent;

    program_node& input() const {
        return get_dependency(0);
    }

    bool generates_dynamic_output() const override {
        return true;
    }
};

using unique_node = typed_program_node<unique>;

template <>
class typed_primitive_inst<unique> : public typed_primitive_inst_base<unique> {
public:
    using parent = typed_primitive_inst_base<unique>;
    using parent::parent;

    static layout calc_output_layout(const unique_node& node, const kernel_impl_params& impl_param);
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const unique_node& node, const kernel_impl_params& impl_param);
    static std::string to_string(const unique_node& node);
};

using unique_inst = typed_primitive_inst<unique>;

}  // namespace cldnn
