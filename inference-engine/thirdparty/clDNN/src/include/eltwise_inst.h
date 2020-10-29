/*
// Copyright (c) 2016-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/eltwise.hpp"
#include "primitive_inst.h"
#include <memory>
#include "topology_impl.h"
#include "kernel_selector/core/actual_kernels/eltwise/eltwise_kernel_base.h"
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

template <>
struct typed_program_node<eltwise> : public typed_program_node_base<eltwise> {
    using parent = typed_program_node_base<eltwise>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program_impl& prog)
        : parent(prim, prog) {
        support_padding_all(true);
    }

    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    size_t inputs_count() const { return get_primitive()->input.size(); }

    std::shared_ptr<kernel_selector::fuse_params> get_fuse_params() const override {
        kernel_selector::eltwise_mode mode = convert_to_eltwise_mode(get_primitive()->mode);
        return std::make_shared<kernel_selector::eltwise_fuse_params>(mode);
    }
};

using eltwise_node = typed_program_node<eltwise>;

template <>
class typed_primitive_inst<eltwise> : public typed_primitive_inst_base<eltwise> {
    using parent = typed_primitive_inst_base<eltwise>;
    static void check_inputs_count(eltwise_node const& node);

public:
    static layout calc_output_layout(eltwise_node const& node);
    static std::string to_string(eltwise_node const& node);

public:
    typed_primitive_inst(network_impl& network, eltwise_node const& node);
};

using eltwise_inst = typed_primitive_inst<eltwise>;

}  // namespace cldnn
