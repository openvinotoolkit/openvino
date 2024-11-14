// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/non_zero.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<count_nonzero> : public typed_program_node_base<count_nonzero> {
    using parent = typed_program_node_base<count_nonzero>;
    typed_program_node(const std::shared_ptr<count_nonzero> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }
public:
    using parent::parent;

    program_node& input() const {
        OPENVINO_ASSERT(dependencies.size() == 1, "[GPU] Primitive ", id(), " has invalid number of depndencies");
        return get_dependency(0);
    }
};

using count_nonzero_node = typed_program_node<count_nonzero>;

template <>
class typed_primitive_inst<count_nonzero> : public typed_primitive_inst_base<count_nonzero> {
    using parent = typed_primitive_inst_base<count_nonzero>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(count_nonzero_node const& /*node*/, kernel_impl_params const& impl_param);
    static layout calc_output_layout(count_nonzero_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(count_nonzero_node const& node);

    typed_primitive_inst(network& network, count_nonzero_node const& node);
};

using count_nonzero_inst = typed_primitive_inst<count_nonzero>;

template <>
struct typed_program_node<gather_nonzero> : public typed_program_node_base<gather_nonzero> {
    using parent = typed_program_node_base<gather_nonzero>;
    typed_program_node(const std::shared_ptr<gather_nonzero> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    program_node& input(size_t index = 0) const {
        OPENVINO_ASSERT(dependencies.size() == 2, "[GPU] Primitive ", id(), " has invalid number of depndencies");
        return get_dependency(index);
    }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {1}; }
};

using gather_nonzero_node = typed_program_node<gather_nonzero>;

template <>
class typed_primitive_inst<gather_nonzero> : public typed_primitive_inst_base<gather_nonzero> {
    using parent = typed_primitive_inst_base<gather_nonzero>;
    using parent::parent;

public:
    template <typename ShapeType>
    static std::vector<layout> calc_output_layouts(gather_nonzero_node const& /*node*/, kernel_impl_params const& impl_param);
    static layout calc_output_layout(gather_nonzero_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(gather_nonzero_node const& node);

    typed_primitive_inst(network& network, gather_nonzero_node const& node);
};

using gather_nonzero_inst = typed_primitive_inst<gather_nonzero>;
}  // namespace cldnn
