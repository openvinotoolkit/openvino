// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/convolution.hpp"
#include "primitive_inst.h"
#include "intel_gpu/runtime/format.hpp"

#include <memory>
#include <string>
#include <vector>

namespace cldnn {

template <>
struct typed_program_node<convolution> : public typed_program_node_base<convolution> {
    using parent = typed_program_node_base<convolution>;

public:
    typed_program_node(std::shared_ptr<primitive> prim, program& prog)
        : parent(prim, prog),
          split(this->get_primitive()->split()),
          depthwise_sep_opt(false),
          transposed(false),
          groups(this->get_primitive()->groups),
          deformable_groups(this->get_primitive()->deformable_groups),
          deformable_mode(this->get_primitive()->deformable_mode) {
        support_padding_all(true);
    }

    void set_split(int32_t node_split) { split = node_split; }
    int32_t get_split() const { return split; }

    void set_depthwise_sep_opt(bool node_depthwise_sep_opt) { depthwise_sep_opt = node_depthwise_sep_opt; }
    bool get_depthwise_sep_opt() const { return depthwise_sep_opt; }

    void set_transposed(bool node_transposed) { transposed = node_transposed; }
    bool get_transposed() const { return transposed; }

    void set_groups(uint32_t node_groups) { groups = node_groups; }
    uint32_t get_groups() const { return groups; }

    void set_deformable_groups(uint32_t node_deformable_groups) { deformable_groups = node_deformable_groups; }
    uint32_t get_deformable_groups() const { return deformable_groups; }

    int32_t get_deform_conv_dep_offset() const {
        auto offset = deformable_mode ? 1 : 0;
        if (get_primitive()->input.size() == 3)
            offset++;
        return offset;
    }

    program_node& input() const { return get_dependency(0); }

    program_node& weights(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("weights offset too big");

        return get_dependency(1 + idx + get_deform_conv_dep_offset());
    }

    program_node& bias(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("bias offset too big");

        return get_dependency(1 + this->get_split() + idx + get_deform_conv_dep_offset());
    }

    program_node& weights_zero_points(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("weights zero points offset too big");

        return get_dependency(1 + (1 + 1 * bias_term()) * this->get_split() + idx + get_deform_conv_dep_offset());
    }

    program_node& activations_zero_points(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("activations zero points offset too big");

        return get_dependency(1 + (1 + 1 * bias_term() + 1 * weights_zero_points_term()) * this->get_split() + idx +
                              get_deform_conv_dep_offset());
    }

    program_node& compensation(size_t idx = 0) const {
        if (static_cast<int32_t>(idx) >= this->get_split())
            throw std::range_error("activations zero points offset too big");

        return get_dependency(1 + (1 + 1 * bias_term() + 1 * weights_zero_points_term() + 1*activations_zero_points_term()) * this->get_split()
                              + idx + get_deform_conv_dep_offset());
    }

    program_node& trans() const {
        if (!deformable_mode)
            throw std::range_error("trans input exists only in deformable mode");

        return get_dependency(1);
    }

    program_node& mask() const {
        if (!deformable_mode)
            throw std::range_error("Mask input exists only in deformable mode");

        return get_dependency(2);
    }

    bool bilinear_interpolation_pad() const {
        if (!deformable_mode)
            throw std::range_error("bilinear_interpolation_pad exists only in deformable mode");
        return get_primitive()->bilinear_interpolation_pad;
    }

    bool get_deformable_mode() const {
        return deformable_mode;
    }

    bool bias_term() const { return get_primitive()->bias.size() > 0; }
    bool weights_zero_points_term() const { return get_primitive()->weights_zero_points.size() > 0; }
    bool compensation_term() const { return get_primitive()->compensation.size() > 0; }
    bool activations_zero_points_term() const { return get_primitive()->activations_zero_points.size() > 0; }

    using parent::get_kernel_impl_params;
    std::unique_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts, const layout& out_layout) const override {
        auto params = parent::get_kernel_impl_params(in_layouts, out_layout);
        params->weights_layout = optional_layout(weights().get_output_layout());
        if (bias_term())
            params->bias_layout = optional_layout(bias().get_output_layout());
        if (weights_zero_points_term())
            params->weights_zero_points_layout = optional_layout(weights_zero_points().get_output_layout());
        if (activations_zero_points_term())
            params->activations_zero_points_layout = optional_layout(activations_zero_points().get_output_layout());
        if (compensation_term())
            params->compensation_layout = optional_layout(compensation().get_output_layout());
        return params;
    }

private:
    int32_t split;
    bool depthwise_sep_opt;
    bool transposed;
    uint32_t groups;
    uint32_t deformable_groups;
    bool deformable_mode;
};

using convolution_node = typed_program_node<convolution>;

template <>
class typed_primitive_inst<convolution> : public typed_primitive_inst_base<convolution> {
    using parent = typed_primitive_inst_base<convolution>;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(convolution_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(convolution_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(convolution_node const& node);

public:
    typed_primitive_inst(network& network, convolution_node const& node);

    memory::ptr weights_memory(size_t index) const {
        if (is_dynamic() && _impl_params->reordered_weights != nullptr) {
            return _impl_params->reordered_weights;
        } else if (_groups == 1) {
            if (static_cast<int32_t>(index) >= _split)
                throw std::range_error("weights offset too big");
            return dep_memory_ptr(1 + index + _deform_conv_dep_offset);
        } else {  // all weights are in one buffer
            return dep_memory_ptr(1 + _deform_conv_dep_offset);
        }
    }

    memory::ptr bias_memory(size_t index) const {
        if (_groups == 1) {
            if (static_cast<int32_t>(index) >= _split)
                throw std::range_error("bias offset too big");
            return dep_memory_ptr(1 + _split + index + _deform_conv_dep_offset);
        } else {  // all bias are in one buffer
            return dep_memory_ptr(2 + _deform_conv_dep_offset);
        }
    }

    memory::ptr weights_zero_points_memory(size_t) const {
        if (_split > 1)
            throw std::range_error("Split is unsupported for quantized convolutions");
        return dep_memory_ptr(2 + 1 * bias_term() + _deform_conv_dep_offset);
    }

    memory::ptr trans_memory() const {
        if (_deform_conv_dep_offset == 0)
            throw std::range_error("trans input exists only in deformable mode");
        return dep_memory_ptr(1);
    }

    memory::ptr activations_zero_points_memory(size_t) const {
        if (_split > 1)
            throw std::range_error("Split is unsupported for quantized convolutions");
        return dep_memory_ptr(2 + 1 * bias_term() + 1 * weights_zero_points_term()
                              + _deform_conv_dep_offset);
    }

    memory::ptr compensation_memory(size_t) const {
        if (_split > 1)
            throw std::range_error("Split is unsupported for quantized convolutions");
        return dep_memory_ptr(2 + 1 * bias_term()
                              + 1 * weights_zero_points_term()
                              + 1 * activations_zero_points_term()
                              + _deform_conv_dep_offset);
    }

    bool bias_term() const { return _impl_params->bias_layout.has_value(); }

    bool weights_zero_points_term() const { return _impl_params->weights_zero_points_layout.has_value(); }
    bool compensation_term() const { return _impl_params->compensation_layout.has_value(); }
    bool activations_zero_points_term() const { return _impl_params->activations_zero_points_layout.has_value(); }

private:
    uint32_t _groups;
    int32_t _split;
    int32_t _deform_conv_dep_offset;
};

using convolution_inst = typed_primitive_inst<convolution>;

}  // namespace cldnn
