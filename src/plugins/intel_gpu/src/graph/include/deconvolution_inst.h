// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/deconvolution.hpp"
#include "primitive_inst.h"
#include "intel_gpu/runtime/format.hpp"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<deconvolution> : public typed_program_node_base<deconvolution> {
    using parent = typed_program_node_base<deconvolution>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(prim, prog),
          groups(this->get_primitive()->groups) {
        support_padding_all(true);
    }

    uint32_t get_groups() const { return groups; }

    program_node& input() const { return get_dependency(0); }
    program_node& weights() const { return get_dependency(1);}
    program_node& bias() const { return get_dependency(2);}

    bool bias_term() const { return !get_primitive()->bias.empty();}

    std::vector<size_t> get_shape_infer_dependencies() const override { return {2}; }

    using parent::get_kernel_impl_params;
    std::unique_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts, const std::vector<layout>& out_layouts) const override {
        auto params = parent::get_kernel_impl_params(in_layouts, out_layouts);
        params->weights_layout = optional_layout(weights().get_output_layout());
        if (bias_term())
            params->bias_layout = optional_layout(bias().get_output_layout());
        return params;
    }

private:
    uint32_t groups;
};

using deconvolution_node = typed_program_node<deconvolution>;

template <>
class typed_primitive_inst<deconvolution> : public typed_primitive_inst_base<deconvolution> {
    using parent = typed_primitive_inst_base<deconvolution>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(deconvolution_node const& node, const kernel_impl_params& impl_param);
    static layout calc_output_layout(deconvolution_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(deconvolution_node const& node);

    bool need_reset_input_memory(size_t idx = 0) const override {
        if (idx != 0)
            return false;

        auto input_layout = _deps[0].first->_impl_params->get_output_layout(0);
        return input_layout.data_padding ? true : false;
    }

    bool need_reset_output_memory() const override {
        bool res = parent::need_reset_output_memory();
        auto output_layout = _impl_params->get_output_layout(0);
        if (output_layout.data_padding) {
            return true;
        }
        return res;
    }

    typed_primitive_inst(network& network, deconvolution_node const& node);

    memory::ptr weights_memory() const {
        if (is_dynamic()) {
            auto weights_mem = _reordered_weights_cache.get(*_impl_params->weights_layout);
            OPENVINO_ASSERT(weights_mem != nullptr, "[GPU] Can't find proper weights memory buffer in cache");
            return weights_mem;
        } else {
            return dep_memory_ptr(1);
        }
    }

    memory::ptr bias_memory() const { return dep_memory_ptr(2); }

    bool bias_term() const { return _impl_params->bias_layout.has_value(); }
};

using deconvolution_inst = typed_primitive_inst<deconvolution>;

}  // namespace cldnn
