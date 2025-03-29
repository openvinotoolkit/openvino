// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/unique.hpp"
#include "primitive_inst.h"

namespace cldnn {

template <>
struct typed_program_node<unique_count> : typed_program_node_base<unique_count> {
    using parent = typed_program_node_base<unique_count>;
    using parent::parent;

    program_node& input() const {
        return get_dependency(0);
    }
};

using unique_count_node = typed_program_node<unique_count>;

template <>
class typed_primitive_inst<unique_count> : public typed_primitive_inst_base<unique_count> {
public:
    using parent = typed_primitive_inst_base<unique_count>;
    using parent::parent;

    static layout calc_output_layout(const unique_count_node& node, const kernel_impl_params& impl_param);
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const unique_count_node& node, const kernel_impl_params& impl_param);
    static std::string to_string(const unique_count_node& node);
};

using unique_count_inst = typed_primitive_inst<unique_count>;

template <>
struct typed_program_node<unique_gather> : typed_program_node_base<unique_gather> {
    using parent = typed_program_node_base<unique_gather>;
    using parent::parent;

    program_node& input() const {
        return get_dependency(0);
    }

    std::vector<size_t> get_shape_infer_dependencies() const override {
        return {1};
    }
};

using unique_gather_node = typed_program_node<unique_gather>;

template <>
class typed_primitive_inst<unique_gather> : public typed_primitive_inst_base<unique_gather> {
public:
    using parent = typed_primitive_inst_base<unique_gather>;
    using parent::parent;

    static layout calc_output_layout(const unique_gather_node& node, const kernel_impl_params& impl_param);
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(const unique_gather_node& node,
                                                   const kernel_impl_params& impl_param);
    static std::string to_string(const unique_gather_node& node);
};

using unique_gather_inst = typed_primitive_inst<unique_gather>;

}  // namespace cldnn
