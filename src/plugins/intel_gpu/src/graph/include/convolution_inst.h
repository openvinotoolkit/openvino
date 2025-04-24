// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
          groups(this->get_primitive()->groups),
          deformable_groups(this->get_primitive()->deformable_groups),
          deformable_mode(this->get_primitive()->deformable_mode) {
        support_padding_all(true);
    }

    bool get_transposed() const { return get_primitive()->transposed; }

    uint32_t get_groups() const { return groups; }

    uint32_t get_deformable_groups() const { return deformable_groups; }

    int32_t get_deform_conv_dep_offset() const {
        auto offset = deformable_mode ? 1 : 0;
        if (get_primitive()->input.size() == 3)
            offset++;
        return offset;
    }

    program_node& input() const { return get_dependency(0); }

    program_node& weights() const {
        return get_dependency(1 + get_deform_conv_dep_offset());
    }

    program_node& bias() const {
        return get_dependency(2 + get_deform_conv_dep_offset());
    }

    program_node& weights_zero_points() const {
        return get_dependency(2 + (1 * bias_term()) + get_deform_conv_dep_offset());
    }

    program_node& activations_zero_points() const {
        return get_dependency(2 + (1 * bias_term() + 1 * weights_zero_points_term()) + get_deform_conv_dep_offset());
    }

    program_node& compensation() const {
        return get_dependency(2 + (1 * bias_term() + 1 * weights_zero_points_term() + 1*activations_zero_points_term()) + get_deform_conv_dep_offset());
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
    bool use_explicit_padding() const { return get_primitive()->auto_pad == ov::op::PadType::EXPLICIT; }

    // Currently convolution with constant weight is only supported for dynamic shape
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }

    using parent::get_kernel_impl_params;
    std::unique_ptr<kernel_impl_params> get_kernel_impl_params(const std::vector<layout>& in_layouts, const std::vector<layout>& out_layouts) const override {
        auto params = parent::get_kernel_impl_params(in_layouts, out_layouts);
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
    uint32_t groups;
    uint32_t deformable_groups;
    bool deformable_mode;
};

using convolution_node = typed_program_node<convolution>;

template <>
class typed_primitive_inst<convolution> : public typed_primitive_inst_base<convolution> {
    using parent = typed_primitive_inst_base<convolution>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(convolution_node const& node, kernel_impl_params const& impl_param);
    static layout calc_output_layout(convolution_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(convolution_node const& node);

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

    typed_primitive_inst(network& network, convolution_node const& node);

    memory::ptr weights_memory() const {
        if (is_dynamic()) {
            auto weights_mem = _reordered_weights_cache.get(*_impl_params->weights_layout);
            OPENVINO_ASSERT(weights_mem != nullptr, "[GPU] Can't find proper weights memory buffer in cache");
            return weights_mem;
        } else {  // all weights are in one buffer
            return dep_memory_ptr(1 + _deform_conv_dep_offset);
        }
    }

    memory::ptr bias_memory() const {
        return dep_memory_ptr(2 + _deform_conv_dep_offset);
    }

    memory::ptr weights_zero_points_memory() const {
        return dep_memory_ptr(2 + 1 * bias_term() + _deform_conv_dep_offset);
    }

    memory::ptr trans_memory() const {
        if (_deform_conv_dep_offset == 0)
            throw std::range_error("trans input exists only in deformable mode");
        return dep_memory_ptr(1);
    }

    memory::ptr activations_zero_points_memory() const {
        return dep_memory_ptr(2 + 1 * bias_term() + 1 * weights_zero_points_term()
                              + _deform_conv_dep_offset);
    }

    memory::ptr compensation_memory() const {
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
    int32_t _deform_conv_dep_offset = 0;
};

using convolution_inst = typed_primitive_inst<convolution>;

}  // namespace cldnn
