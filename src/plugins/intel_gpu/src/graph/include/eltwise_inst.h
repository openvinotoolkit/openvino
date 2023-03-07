// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/eltwise.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>

namespace cldnn {

inline kernel_selector::eltwise_mode convert_to_eltwise_mode(eltwise_mode mode) {
    switch (mode) {
        case eltwise_mode::sum:
            return kernel_selector::eltwise_mode::ADD;
        case eltwise_mode::sub:
            return kernel_selector::eltwise_mode::SUB;
        case eltwise_mode::max:
            return kernel_selector::eltwise_mode::MAX;
        case eltwise_mode::prod:
            return kernel_selector::eltwise_mode::MUL;
        case eltwise_mode::div:
            return kernel_selector::eltwise_mode::DIV;
        case eltwise_mode::min:
            return kernel_selector::eltwise_mode::MIN;
        case eltwise_mode::pow:
            return kernel_selector::eltwise_mode::POW;
        case eltwise_mode::mod:
            return kernel_selector::eltwise_mode::MODULU;
        case eltwise_mode::eq:
            return kernel_selector::eltwise_mode::EQ;
        case eltwise_mode::ne:
            return kernel_selector::eltwise_mode::NE;
        case eltwise_mode::lt:
            return kernel_selector::eltwise_mode::LT;
        case eltwise_mode::le:
            return kernel_selector::eltwise_mode::LE;
        case eltwise_mode::gt:
            return kernel_selector::eltwise_mode::GT;
        case eltwise_mode::ge:
            return kernel_selector::eltwise_mode::GE;
        case eltwise_mode::logic_and:
            return kernel_selector::eltwise_mode::LOGIC_AND;
        case eltwise_mode::logic_or:
            return kernel_selector::eltwise_mode::LOGIC_OR;
        case eltwise_mode::logic_xor:
            return kernel_selector::eltwise_mode::LOGIC_XOR;
        case eltwise_mode::squared_diff:
            return kernel_selector::eltwise_mode::SQUARED_DIFF;
        case eltwise_mode::floor_mod:
            return kernel_selector::eltwise_mode::FLOOR_MOD;
        default:
            return kernel_selector::eltwise_mode::ADD;
    }
}

class EltwiseFuseParams : public NodeFuseParams {
public:
    EltwiseFuseParams(std::shared_ptr<eltwise> desc) : NodeFuseParams(eltwise::type_id()), _desc(desc) {}
    size_t ops_count() const override { return 1; }

    std::shared_ptr<eltwise> _desc;
};

template <>
struct typed_program_node<eltwise> : public typed_program_node_base<eltwise> {
    using parent = typed_program_node_base<eltwise>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(prim, prog) {
        support_padding_all(true);
    }

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    size_t inputs_count() const { return get_primitive()->input.size(); }

    std::shared_ptr<NodeFuseParams> get_fuse_params() const override {
        return std::make_shared<EltwiseFuseParams>(typed_desc());
    }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using eltwise_node = typed_program_node<eltwise>;

template <>
class typed_primitive_inst<eltwise> : public typed_primitive_inst_base<eltwise> {
    using parent = typed_primitive_inst_base<eltwise>;
    using parent::parent;
    static void check_inputs_count(eltwise_node const& node);

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(eltwise_node const& /*node*/, const kernel_impl_params& impl_param);
    static layout calc_output_layout(eltwise_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(eltwise_node const& node);
    typed_primitive_inst(network& network, eltwise_node const& node);
};

using eltwise_inst = typed_primitive_inst<eltwise>;

}  // namespace cldnn
