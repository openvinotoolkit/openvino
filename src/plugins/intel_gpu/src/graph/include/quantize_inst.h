// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/quantize.hpp"
#include "primitive_inst.h"
#include "data_inst.h"
#include <string>
#include <memory>

namespace cldnn {
class QuantizeFuseParams : public NodeFuseParams {
public:
    QuantizeFuseParams(const layout& out_layout,
                       bool scale_shift_opt,
                       bool need_post_scale,
                       bool need_post_shift,
                       bool need_pre_shift,
                       bool need_clamp,
                       bool need_min_clamp,
                       bool need_max_clamp,
                       bool per_tensor_input_range,
                       bool per_tensor_input_scale,
                       bool per_tensor_input_shift,
                       bool per_tensor_output_range,
                       bool per_tensor_output_scale,
                       bool per_tensor_output_shift,
                       float in_lo,
                       float in_hi,
                       float in_scale,
                       float in_shift,
                       float out_lo,
                       float out_hi,
                       float out_scale,
                       float out_shift)
    : NodeFuseParams(quantize::type_id())
    , _out_layout(out_layout)
    , _scale_shift_opt(scale_shift_opt)
    , _need_post_scale(need_post_scale)
    , _need_post_shift(need_post_shift)
    , _need_pre_shift(need_pre_shift)
    , _need_clamp(need_clamp)
    , _need_min_clamp(need_min_clamp)
    , _need_max_clamp(need_max_clamp)
    , _need_out_range(true)
    , _per_tensor_input_range(per_tensor_input_range)
    , _per_tensor_input_scale(per_tensor_input_scale)
    , _per_tensor_input_shift(per_tensor_input_shift)
    , _per_tensor_output_range(per_tensor_output_range)
    , _per_tensor_output_scale(per_tensor_output_scale)
    , _per_tensor_output_shift(per_tensor_output_shift)
    , _in_lo(in_lo)
    , _in_hi(in_hi)
    , _in_scale(in_scale)
    , _in_shift(in_shift)
    , _out_lo(out_lo)
    , _out_hi(out_hi)
    , _out_scale(out_scale)
    , _out_shift(out_shift)  {
        size_t index = 1; // skip activations input
        _need_out_range = _per_tensor_output_range && _out_lo < _out_hi;
        if (!_need_out_range && _need_clamp) {
            in_range_lo_idx = index++;
            in_range_hi_idx = index++;
        }
        if (!_per_tensor_input_scale) {
            in_scale_idx = index++;
        }
        if (!_per_tensor_input_shift && _need_pre_shift) {
            in_shift_idx = index++;
        }
        if (!_per_tensor_output_scale && _need_post_scale) {
            out_scale_idx = index++;
        }
        if (!_per_tensor_output_shift && _need_post_shift) {
            out_shift_idx = index++;
        }
    }

    size_t ops_count() const override {
        size_t count = 0;
        // pre-scale, pre-shift
        if (_per_tensor_input_scale && _per_tensor_input_shift) {
            count++;
        } else {
            count += 2;
        }

        // post-scale, post-shift
        if (_need_post_scale && _need_post_shift && _per_tensor_output_scale && _per_tensor_output_shift) {
            count++;
        } else {
            count += 2;
        }

        auto out_dt = _out_layout.data_type;
        auto output_type_is_int8 = out_dt == data_types::u8 || out_dt == data_types::i8;
        auto out_range_usage = _per_tensor_output_range && (_out_lo < _out_hi);

        if (out_range_usage) {
            // round
            if (!output_type_is_int8) {
                count++;
            }

            // clamp
            if (_need_clamp) {
                count++;
            }
        } else {
            // clamp
            if (_need_clamp) {
                count += 2;
            }
            // round
            {
                count++;
            }
        }

        return count;
    }

    layout _out_layout;

    bool _scale_shift_opt;
    bool _need_post_scale;
    bool _need_post_shift;
    bool _need_pre_shift;
    bool _need_clamp;
    bool _need_min_clamp;
    bool _need_max_clamp;
    bool _need_out_range;

    bool _per_tensor_input_range;
    bool _per_tensor_input_scale;
    bool _per_tensor_input_shift;
    bool _per_tensor_output_range;
    bool _per_tensor_output_scale;
    bool _per_tensor_output_shift;

    float _in_lo;
    float _in_hi;
    float _in_scale;
    float _in_shift;
    float _out_lo;
    float _out_hi;
    float _out_scale;
    float _out_shift;

    size_t in_range_lo_idx = 0;
    size_t in_range_hi_idx = 0;
    size_t in_scale_idx = 0;
    size_t in_shift_idx = 0;
    size_t out_scale_idx = 0;
    size_t out_shift_idx = 0;
};

template <>
struct typed_program_node<quantize> : public typed_program_node_base<quantize> {
    using parent = typed_program_node_base<quantize>;

    typed_program_node(std::shared_ptr<quantize> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

public:
    using parent::parent;

    program_node& input(size_t index = 0) const { return get_dependency(index); }
    int get_levels() const { return get_primitive()->levels; }
    bool get_scale_shift_opt() const { return get_primitive()->scale_shift_opt; }
    bool get_need_pre_shift() const { return get_primitive()->need_pre_shift; }
    bool get_need_post_scale() const { return get_primitive()->need_post_scale; }
    bool get_need_post_shift() const { return get_primitive()->need_post_shift; }
    bool get_need_clamp() const { return get_primitive()->need_clamp; }
    bool get_need_min_clamp() const { return get_primitive()->need_min_clamp; }
    bool get_need_max_clamp() const { return get_primitive()->need_max_clamp; }
    bool get_per_tensor_input_scale() const { return get_primitive()->per_tensor_input_scale; }
    bool get_per_tensor_input_shift() const { return get_primitive()->per_tensor_input_shift; }
    bool get_per_tensor_input_range() const { return get_primitive()->per_tensor_input_range; }
    bool get_per_tensor_output_scale() const { return get_primitive()->per_tensor_output_scale; }
    bool get_per_tensor_output_shift() const { return get_primitive()->per_tensor_output_shift; }
    bool get_per_tensor_output_range() const { return get_primitive()->per_tensor_output_range; }
    float get_input_scale_val() const { return get_primitive()->in_scale; }
    float get_input_shift_val() const { return get_primitive()->in_shift; }
    float get_input_lo_val() const { return get_primitive()->in_lo; }
    float get_input_hi_val() const { return get_primitive()->in_hi; }
    float get_output_scale_val() const { return get_primitive()->out_scale; }
    float get_output_shift_val() const { return get_primitive()->out_shift; }
    float get_output_lo_val() const { return get_primitive()->out_lo; }
    float get_output_hi_val() const { return get_primitive()->out_hi; }

    std::shared_ptr<NodeFuseParams> get_fuse_params() const override {
        return std::make_shared<QuantizeFuseParams>(get_output_layout(),
                                                    get_primitive()->scale_shift_opt,
                                                    get_primitive()->need_post_scale,
                                                    get_primitive()->need_post_shift,
                                                    get_primitive()->need_pre_shift,
                                                    get_primitive()->need_clamp,
                                                    get_primitive()->need_min_clamp,
                                                    get_primitive()->need_max_clamp,
                                                    get_primitive()->per_tensor_input_range,
                                                    get_primitive()->per_tensor_input_scale,
                                                    get_primitive()->per_tensor_input_shift,
                                                    get_primitive()->per_tensor_output_range,
                                                    get_primitive()->per_tensor_output_scale,
                                                    get_primitive()->per_tensor_output_shift,
                                                    get_primitive()->in_lo,
                                                    get_primitive()->in_hi,
                                                    get_primitive()->in_scale,
                                                    get_primitive()->in_shift,
                                                    get_primitive()->out_lo,
                                                    get_primitive()->out_hi,
                                                    get_primitive()->out_scale,
                                                    get_primitive()->out_shift);
    }
    std::vector<size_t> get_shape_infer_dependencies() const override { return {}; }
};

using quantize_node = typed_program_node<quantize>;

template <>
class typed_primitive_inst<quantize> : public typed_primitive_inst_base<quantize> {
    using parent = typed_primitive_inst_base<quantize>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(quantize_node const& node, kernel_impl_params const& impl_param) {
        return forward_input0_shape<ShapeType>(impl_param);
    }
    static layout calc_output_layout(quantize_node const& node, kernel_impl_params const& impl_param);
    static std::string to_string(quantize_node const& node);

    typed_primitive_inst(network& network, quantize_node const& node);
};

using quantize_inst = typed_primitive_inst<quantize>;

}  // namespace cldnn
