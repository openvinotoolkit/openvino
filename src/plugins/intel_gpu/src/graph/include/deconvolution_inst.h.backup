// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "intel_gpu/primitives/deconvolution.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<deconvolution> : public typed_program_node_base<deconvolution> {
    using parent = typed_program_node_base<deconvolution>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(prim, prog),
          split(this->get_primitive()->split()),
          depthwise_sep_opt(false),
          groups(this->get_primitive()->groups) {
        support_padding_all(true);
    }

    void set_split(int32_t node_split) { split = node_split; }
    int32_t get_split() const { return split; }

    void set_depthwise_sep_opt(bool node_depthwise_sep_opt) { depthwise_sep_opt = node_depthwise_sep_opt; }
    bool get_depthwise_sep_opt() const { return depthwise_sep_opt; }

    void set_groups(uint32_t node_groups) { groups = node_groups; }
    uint32_t get_groups() const { return groups; }

    program_node& input() const { return get_dependency(0); }

    program_node& weights(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= get_split())
            throw std::range_error("weights offset too big");

        return get_dependency(1 + idx);
    }

    program_node& bias(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= get_split())
            throw std::range_error("bias offset too big");

        return get_dependency(1 + this->get_split() + idx);
    }

    bool bias_term() const {
        if (get_primitive()->bias.size() != 0)
            return true;
        else
            return false;
    }

    program_node& fused_sum(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) > 0)
            throw std::range_error("Only one input for fused sum is supported");

        size_t d_idx = 1 + this->get_split() + idx;
        d_idx += bias_term() ? this->get_split() : 0;
        return get_dependency(d_idx);
    }

    bool has_fused_sum() const {
        size_t d_idx = 1 + this->get_split();
        d_idx += bias_term() ? this->get_split() : 0;
        return dependencies.size() == (d_idx + 1);
    }

private:
    int32_t split;
    bool depthwise_sep_opt;
    uint32_t groups;
};

using deconvolution_node = typed_program_node<deconvolution>;

template <>
class typed_primitive_inst<deconvolution> : public typed_primitive_inst_base<deconvolution> {
    using parent = typed_primitive_inst_base<deconvolution>;

public:
    static layout calc_output_layout(deconvolution_node const& node);
    static std::string to_string(deconvolution_node const& node);

public:
    typed_primitive_inst(network& network, deconvolution_node const& node);

    memory::ptr weights_memory(size_t index) const {
        if (node.get_groups() == 1) {
            if (static_cast<int32_t>(index) >= node.get_split())
                throw std::range_error("weights offset too big");
            return dep_memory_ptr(1 + index);
        } else {  // all weights are in one buffer
            return dep_memory_ptr(1);
        }
    }

    memory::ptr bias_memory(size_t index) const {
        if (node.get_groups() == 1) {
            if (argument.bias.size() == 0 && static_cast<int32_t>(index) >= node.get_split())
                throw std::range_error("no bias data");
            if (static_cast<int32_t>(index) > node.get_split())
                throw std::range_error("bias offset too big");
            return dep_memory_ptr(1 + node.get_split() + index);
        } else {  // all bias are in one buffer
            return dep_memory_ptr(2);
        }
    }

    bool bias_term() const {
        if (argument.bias.size() != 0)
            return true;
        else
            return false;
    }
};

using deconvolution_inst = typed_primitive_inst<deconvolution>;

}  // namespace cldnn
