// Copyright (c) 2016-2018 Intel Corporation
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

#include "kernel_selector_helper.h"

kernel_selector::data_type to_data_type(data_types dt)
{
    switch (dt)
    {
    case cldnn::data_types::i8:     return kernel_selector::data_type::INT8;
    case cldnn::data_types::u8:     return kernel_selector::data_type::UINT8;
    case cldnn::data_types::i32:     return kernel_selector::data_type::INT32;
    case cldnn::data_types::i64:     return kernel_selector::data_type::INT64;
    case cldnn::data_types::f16:    return kernel_selector::data_type::F16;
    case cldnn::data_types::f32:    return kernel_selector::data_type::F32;
    default:
        assert(0);
        return kernel_selector::data_type::F16;
    }
}

data_types from_data_type(kernel_selector::data_type dt)
{
    switch (dt)
    {
    case kernel_selector::data_type::INT8:   return cldnn::data_types::i8;
    case kernel_selector::data_type::UINT8:   return cldnn::data_types::u8;
    case kernel_selector::data_type::INT32:   return cldnn::data_types::i32;
    case kernel_selector::data_type::INT64:   return cldnn::data_types::i64;
    case kernel_selector::data_type::F16:    return cldnn::data_types::f16;
    case kernel_selector::data_type::F32:    return cldnn::data_types::f32;
    default:
        assert(0);
        return cldnn::data_types::f16;
    }
}

kernel_selector::weights_type to_weights_type(data_types dt)
{
    switch (dt)
    {
    case cldnn::data_types::i8:     return kernel_selector::weights_type::INT8;
    case cldnn::data_types::f16:    return kernel_selector::weights_type::F16;
    case cldnn::data_types::f32:    return kernel_selector::weights_type::F32;
    default:
        assert(0);
        return kernel_selector::weights_type::F16;
    }
}

data_types from_weights_type(kernel_selector::weights_type dt)
{
    switch (dt)
    {
    case kernel_selector::weights_type::INT8:   return data_types::i8;
    case kernel_selector::weights_type::F16:    return data_types::f16;
    case kernel_selector::weights_type::F32:    return data_types::f32;
    default:
        assert(0);
        return data_types::f16;;
    }
}

kernel_selector::data_layout to_data_layout(format f)
{
    switch (f)
    {
    case format::bfyx:              return kernel_selector::data_layout::bfyx;
    case format::yxfb:              return kernel_selector::data_layout::yxfb;
    case format::byxf:              return kernel_selector::data_layout::byxf;
    case format::fyxb:              return kernel_selector::data_layout::fyxb;
    case format::bs_x_bsv16:        return kernel_selector::data_layout::bs_f_bsv16__af8;
    case format::bs_xs_xsv8_bsv8:   return kernel_selector::data_layout::bs_f_bsv8__af8;
    case format::bs_xs_xsv8_bsv16:  return kernel_selector::data_layout::bs_f_bsv16__af8;
    case format::bf8_xy16:          return kernel_selector::data_layout::bf8_xy16;
    case format::winograd_2x3_s1_data:  return kernel_selector::data_layout::winograd_2x3_s1_data;
    case format::byxf_af32: return kernel_selector::data_layout::byxf_af32;
    case format::fs_bs_yx_bsv4_fsv32: return kernel_selector::data_layout::fs_bs_yx_bsv4_fsv32;
        //     case format::brfyx:          return kernel_selector::data_layout::brfyx;
    default:
        return kernel_selector::data_layout::bfyx;
    }
}

cldnn::format from_data_layout(kernel_selector::data_layout l)
{
    switch (l)
    {
    case kernel_selector::data_layout::bf:                return cldnn::format::bfyx;
    case kernel_selector::data_layout::fb:                return cldnn::format::fyxb;
    case kernel_selector::data_layout::bfyx:              return cldnn::format::bfyx;
    case kernel_selector::data_layout::yxfb:              return cldnn::format::yxfb;
    case kernel_selector::data_layout::byxf:              return cldnn::format::byxf;
    case kernel_selector::data_layout::fyxb:              return cldnn::format::fyxb;
    case kernel_selector::data_layout::bs_f_bsv8__af8:    return cldnn::format::bs_xs_xsv8_bsv8;
    case kernel_selector::data_layout::bs_f_bsv16__af8:   return cldnn::format::bs_x_bsv16;
    case kernel_selector::data_layout::bf8_xy16:          return cldnn::format::bf8_xy16;
    case kernel_selector::data_layout::brfyx:             return cldnn::format::bfyx;
    case kernel_selector::data_layout::winograd_2x3_s1_data:   return cldnn::format::winograd_2x3_s1_data;
    case kernel_selector::data_layout::byxf_af32: return cldnn::format::byxf_af32;
    case kernel_selector::data_layout::fs_bs_yx_bsv4_fsv32: return cldnn::format::fs_bs_yx_bsv4_fsv32;
    default:
        return cldnn::format::bfyx;
        break;
    }
}

kernel_selector::weights_layout to_weights_layout(format f)
{
    switch (f)
    {
    case format::bfyx:              return kernel_selector::weights_layout::oiyx;
    case format::fyxb:              return kernel_selector::weights_layout::iyxo;
    case format::byxf:              return kernel_selector::weights_layout::oyxi;
    case format::yxfb:              return kernel_selector::weights_layout::yxio;
    case format::os_iyx_osv16:      return kernel_selector::weights_layout::os_iyx_osv16;
    case format::bs_xs_xsv8_bsv8:   return kernel_selector::weights_layout::os_i_osv8__ai8;
    case format::bs_xs_xsv8_bsv16:  return kernel_selector::weights_layout::os_i_osv16__ai8;
    case format::bs_x_bsv16:        return kernel_selector::weights_layout::os_i_osv16;
    case format::image_2d_weights_c4_fyx_b:     return kernel_selector::weights_layout::image_2d_weights_c4_fyx_b;
    case format::image_2d_weights_c1_b_fyx:     return kernel_selector::weights_layout::image_2d_weights_c1_b_fyx;
    case format::winograd_2x3_s1_weights:       return kernel_selector::weights_layout::winograd_2x3_s1_weights;
    case format::winograd_2x3_s1_fused_weights: return kernel_selector::weights_layout::winograd_2x3_s1_fused_weights;
    case format::winograd_6x3_s1_fused_weights: return kernel_selector::weights_layout::winograd_6x3_s1_fused_weights;
    case format::image_2d_weights_winograd_6x3_s1_fbxyb:     return kernel_selector::weights_layout::image_2d_weights_winograd_6x3_s1_fbxyb;
    case format::image_2d_weights_winograd_6x3_s1_xfbyb:     return kernel_selector::weights_layout::image_2d_weights_winograd_6x3_s1_xfbyb;
    case format::os_is_yx_isa8_osv8_isv4: return kernel_selector::weights_layout::os_is_yx_isa8_osv8_isv4;
    case format::is_o_yx_isv32: return kernel_selector::weights_layout::is_o_yx_isv32;
    default:
        return kernel_selector::weights_layout::oi;
    }
}

cldnn::format::type from_weights_layout(kernel_selector::weights_layout l)
{
    switch (l)
    {
    case kernel_selector::weights_layout::oi:
    case kernel_selector::weights_layout::oiyx:               return cldnn::format::bfyx;
    case kernel_selector::weights_layout::oyxi:               return cldnn::format::byxf;
    case kernel_selector::weights_layout::io:
    case kernel_selector::weights_layout::iyxo:               return cldnn::format::fyxb;
    case kernel_selector::weights_layout::yxio:               return cldnn::format::yxfb;
    case kernel_selector::weights_layout::os_iyx_osv16:       return cldnn::format::os_iyx_osv16;
    case kernel_selector::weights_layout::os_i_osv16:         return cldnn::format::bs_x_bsv16;
    case kernel_selector::weights_layout::os_i_osv8__ai8:     return cldnn::format::bs_xs_xsv8_bsv8;
    case kernel_selector::weights_layout::os_i_osv16__ai8:    return cldnn::format::bs_xs_xsv8_bsv16;
    case kernel_selector::weights_layout::image_2d_weights_c4_fyx_b:        return cldnn::format::image_2d_weights_c4_fyx_b;
    case kernel_selector::weights_layout::image_2d_weights_c1_b_fyx:        return cldnn::format::image_2d_weights_c1_b_fyx;
    case kernel_selector::weights_layout::winograd_2x3_s1_weights:          return cldnn::format::winograd_2x3_s1_weights;
    case kernel_selector::weights_layout::winograd_2x3_s1_fused_weights:    return cldnn::format::winograd_2x3_s1_fused_weights;
    case kernel_selector::weights_layout::winograd_6x3_s1_fused_weights:    return cldnn::format::winograd_6x3_s1_fused_weights;
    case kernel_selector::weights_layout::image_2d_weights_winograd_6x3_s1_fbxyb:        return cldnn::format::image_2d_weights_winograd_6x3_s1_fbxyb;
    case kernel_selector::weights_layout::image_2d_weights_winograd_6x3_s1_xfbyb:        return cldnn::format::image_2d_weights_winograd_6x3_s1_xfbyb;
    case kernel_selector::weights_layout::os_is_yx_isa8_osv8_isv4: return cldnn::format::os_is_yx_isa8_osv8_isv4;
    case kernel_selector::weights_layout::is_o_yx_isv32: return cldnn::format::is_o_yx_isv32;
    default:
        return cldnn::format::bfyx;
    }
}

kernel_selector::tuning_mode to_tuning_mode(cldnn::tuning_mode mode)
{
    switch (mode)
    {
    case cldnn::tuning_mode::tuning_disabled:         return kernel_selector::tuning_mode::TUNING_DISABLED;
    case cldnn::tuning_mode::tuning_use_cache:        return kernel_selector::tuning_mode::TUNING_USE_CACHE;
    case cldnn::tuning_mode::tuning_tune_and_cache:   return kernel_selector::tuning_mode::TUNING_TUNE_AND_CACHE;
    default:
        return kernel_selector::tuning_mode::TUNING_DISABLED;
    }
}

std::string to_host_version(const cldnn::version_t& version)
{
    std::stringstream ss;
    ss << version.major << "." << version.minor << "." << version.build << "." << version.revision;
    return ss.str();
}

kernel_selector::data_tensor convert_data_tensor(const layout& l, uint32_t split, const tensor view_offset)
{
    const auto& pad = l.data_padding;
    const auto& vals = l.size.sizes(l.format);
    const auto& add_offsets = view_offset.sizes(l.format);
    const auto& lower_pad = pad.lower_size().sizes(l.format);
    const auto& upper_pad = pad.upper_size().sizes(l.format);
    const auto ks_layout = to_data_layout(l.format);
    kernel_selector::n_dims vec(kernel_selector::DataTensor::ChannelsCount(ks_layout));

    size_t pitch = 1;
    size_t offset = 0;

    auto new_vals = vals;

    if (ks_layout == kernel_selector::Tensor::byxf_af32)
    {
        new_vals[3] = align_to(vals[3], 32);
    }
    if (ks_layout == kernel_selector::Tensor::fs_bs_yx_bsv4_fsv32)
    {
        new_vals[3] = align_to(vals[3], 32);
        new_vals[2] = align_to(vals[2], 4);
    }

    for (size_t i = 0; i < vec.size(); i++)
    {
        const size_t tensor_index = vec.size() - 1 - i;
        const auto d = vals[tensor_index];
        const auto lp = lower_pad[tensor_index];
        const auto up = upper_pad[tensor_index];
        // tells us how many elements are reserved in memory for this tensor index
        const auto reserved_in_mem_count = new_vals[tensor_index];

        auto& elm = vec[i];
        elm.v = static_cast<size_t>(d - add_offsets[tensor_index]);
        elm.pitch = pitch;
        elm.pad.before = lp;
        elm.pad.after = up;

        offset += pitch * (add_offsets[tensor_index]);
        pitch *= (reserved_in_mem_count + lp + up);
    }

    const int feature_index = kernel_selector::DataTensor::Channelndex(ks_layout, kernel_selector::Tensor::DataChannelName::FEATURE);
    vec[feature_index].v /= split;

    return kernel_selector::data_tensor(
        vec,
        to_data_type(l.data_type),
        ks_layout,
        offset);
}

kernel_selector::weights_tensor convert_weights_tensor(const layout& l)
{
    assert(l.format.dimension() == 4);
    const auto& t = l.size.sizes(format::bfyx);
    const auto base_layout = kernel_selector::weights_layout::oiyx;
    const auto ks_type = to_weights_type(l.data_type);
    const auto ks_layout = to_weights_layout(l.format);
    std::vector<size_t> vec(kernel_selector::WeightsTensor::ChannelsCount(base_layout));

    for (size_t i = 0; i < vec.size(); i++)
    {
        const size_t tensor_index = t.size() - 1 - i;
        const auto d = t[tensor_index];
        vec[i] = static_cast<size_t>(d);
    }

    return kernel_selector::weights_tensor(
        vec,
        ks_type,
        base_layout).TransformIgnorePadding(ks_layout);
}

kernel_selector::activation_function get_kernel_selector_activation_param(cldnn_activation_func activation_func)
{
    switch (activation_func)
    {
    case activation_none:
        return kernel_selector::activation_function::NONE;
    case activation_logistic:
        return kernel_selector::activation_function::LOGISTIC;
    case activation_hyperbolic_tan:
        return kernel_selector::activation_function::HYPERBOLIC_TAN;
    case activation_relu:
        return kernel_selector::activation_function::RELU;
    case activation_relu_negative_slope:
        return kernel_selector::activation_function::RELU_NEGATIVE_SLOPE;
    case activation_clamp:
        return kernel_selector::activation_function::CLAMP;
    case activation_softrelu:
        return kernel_selector::activation_function::SOFTRELU;
    case activation_abs:
        return kernel_selector::activation_function::ABS;
    case activation_linear:
        return kernel_selector::activation_function::LINEAR;
    case activation_square:
        return kernel_selector::activation_function::SQUARE;
    case activation_sqrt:
        return kernel_selector::activation_function::SQRT;
    case activation_elu:
        return kernel_selector::activation_function::ELU;
    case activation_sin:
        return kernel_selector::activation_function::SIN;
    case activation_asin:
        return kernel_selector::activation_function::ASIN;
    case activation_sinh:
        return kernel_selector::activation_function::SINH;
    case activation_cos:
        return kernel_selector::activation_function::COS;
    case activation_acos:
        return kernel_selector::activation_function::ACOS;
    case activation_cosh:
        return kernel_selector::activation_function::COSH;
    case activation_log:
        return kernel_selector::activation_function::LOG;
	case activation_log2:
		return kernel_selector::activation_function::LOG2;
    case activation_exp:
        return kernel_selector::activation_function::EXP;
    default:
        throw std::runtime_error("Unknown activation function");
        break;
    }
}

kernel_selector::activation_function get_kernel_selector_activation_grad_param(cldnn_activation_grad_func activation_grad_func)
{
    switch (activation_grad_func)
    {
    case activation_grad_none:
        return kernel_selector::activation_function::NONE_GRAD;
    case activation_grad_relu:
        return kernel_selector::activation_function::RELU_GRAD;
    case activation_grad_relu_negative_slope:
        return kernel_selector::activation_function::RELU_NEGATIVE_SLOPE_GRAD;
    default:
        throw std::runtime_error("Unknown activation_grad function");
        break;
    }
}