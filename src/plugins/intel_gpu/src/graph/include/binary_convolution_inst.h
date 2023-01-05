// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/binary_convolution.hpp"
#include "primitive_inst.h"

#include <memory>
#include <string>
#include <vector>

namespace cldnn {

template <>
struct typed_program_node<binary_convolution> : public typed_program_node_base<binary_convolution> {
    using parent = typed_program_node_base<binary_convolution>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(prim, prog), split(this->get_primitive()->split()) {}

    void set_split(int32_t node_split) { split = node_split; }
    int32_t get_split() const { return split; }

    program_node& input() const { return get_dependency(0); }

    program_node& weights(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("weights offset too big");

        return get_dependency(1 + idx);
    }

    // Define bias functions to be able to reuse get_weights_bias_default_params<T> function
    program_node& bias(size_t /*idx*/ = 0) const {
        throw std::runtime_error("bias should not be used in binary convolution");
    }

    std::unique_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts, const std::vector<layout>& out_layouts) const override {
        auto params = parent::get_kernel_impl_params(in_layouts, out_layouts);
        params->weights_layout = optional_layout(weights().get_output_layout());
        return params;
    }

private:
    int32_t split;
};

using binary_convolution_node = typed_program_node<binary_convolution>;

template <>
class typed_primitive_inst<binary_convolution> : public typed_primitive_inst_base<binary_convolution> {
    using parent = typed_primitive_inst_base<binary_convolution>;
    using parent::parent;

public:
    static layout calc_output_layout(binary_convolution_node const& node, kernel_impl_params const& impl_param);

    static std::string to_string(binary_convolution_node const& node);

public:
    typed_primitive_inst(network& network, binary_convolution_node const& node);

    memory::ptr weights_memory(size_t index) const {
        if (static_cast<int32_t>(index) >= node->get_split())
            throw std::range_error("weights offset too big");

        return dep_memory_ptr(1 + index);
    }
};

using binary_convolution_inst = typed_primitive_inst<binary_convolution>;

}  // namespace cldnn
