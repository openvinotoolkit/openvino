// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/activation.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

class ActivationFuseParams : public NodeFuseParams {
public:
    ActivationFuseParams(std::shared_ptr<activation> desc) : NodeFuseParams(activation::type_id()), _desc(desc) {}
    size_t ops_count() const override { return 1; }

    std::shared_ptr<activation> _desc;
};

template <>
struct typed_program_node<activation> : public typed_program_node_base<activation> {
    using parent = typed_program_node_base<activation>;
    typed_program_node(const std::shared_ptr<activation> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    program_node& input() const { return get_dependency(0); }
    program_node& slope_input() const { return get_dependency(1); }

    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }

    bool is_parameterized() const { return !typed_desc()->additional_params_input.empty(); }

    std::shared_ptr<NodeFuseParams> get_fuse_params() const override {
        return std::make_shared<ActivationFuseParams>(typed_desc());
    }
};

using activation_node = typed_program_node<activation>;

template <>
class typed_primitive_inst<activation> : public typed_primitive_inst_base<activation> {
    using parent = typed_primitive_inst_base<activation>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(activation_node const& /*node*/, const kernel_impl_params& impl_param) {
        return forward_input0_shape<ShapeType>(impl_param);
    }

    static layout calc_output_layout(activation_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(activation_node const& node);

    typed_primitive_inst(network& network, activation_node const& node);

    memory::ptr slope_memory() const { return dep_memory_ptr(1); }

    bool is_parameterized() const { return !get_typed_desc<activation>()->additional_params_input.empty(); }
};

using activation_inst = typed_primitive_inst<activation>;
}  // namespace cldnn
