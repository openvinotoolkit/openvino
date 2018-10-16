/*
// Copyright (c) 2016 Intel Corporation
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

#pragma once
#include "api/C/cldnn.h"
#include "api/CPP/program.hpp"
#include "program_impl.h"
#include "gpu/ocl_toolkit.h"
#include "tensor_type.h"
#include "kernel_selector_params.h"
#include "kernel_selector_common.h"
#include "jitter.h"

using namespace cldnn;

namespace kernel_selector
{
    using n_dims                            = kernel_selector::Tensor::NDims;
    using kernel_data                       = kernel_selector::KernelData;
    using kernel_string                     = kernel_selector::KernelString;
    using cl_kernel_data                    = kernel_selector::clKernelData;
    using kernel_arguments                  = kernel_selector::Arguments;
    using kernel_argument_element           = kernel_selector::ArgumentDescriptor;
    using kernel_argument_types             = kernel_selector::ArgumentDescriptor::Types;
    using kernel_scalar_arguments           = kernel_selector::Scalars;
    using kernel_scalar_argument_types      = kernel_selector::ScalarDescriptor::Types;
    using jit_constants                     = kernel_selector::JitConstants;

    using data_type                         = kernel_selector::Datatype;
    using kernel_type                       = kernel_selector::KernelType;
    using weights_type                      = kernel_selector::WeightsType;
    using activation_function               = kernel_selector::ActivationFunction;
    using pool_type                         = kernel_selector::PoolType;
    using pool_remainder                    = kernel_selector::PoolRemainder;
	using argm_axis							= kernel_selector::ArgMaxMinAxis;
	using argm_output						= kernel_selector::ArgMaxMinOut;
    using lookt_axis                        = kernel_selector::LookUpTableAxis;
    using lrn_mode                          = kernel_selector::LRNMode;
    using normalize_mode                    = kernel_selector::NormalizeMode;
    using mvn_mode                          = kernel_selector::MVNMode;
    using kernel_divider_mode               = kernel_selector::KernelDividerMode;
    using eltwise_mode                      = kernel_selector::EltwiseMode;
    using eltwise_input_mode                = kernel_selector::EltwiseInputMode;
    using softmax_dim                       = kernel_selector::SoftmaxDim;
    using mean_subtruct_mode                = kernel_selector::MeanSubtractMode;
    using mean_op                           = kernel_selector::MeanOp;
    using concat_axis                       = kernel_selector::ConcatAxis;
    using tuning_mode                       = kernel_selector::TuningMode;
    using sample_type                       = kernel_selector::SampleType;

    using data_tensor                       = kernel_selector::DataTensor;
    using weights_tensor                    = kernel_selector::WeightsTensor;
    using data_layout                       = kernel_selector::DataLayout;
    using weights_layout                    = kernel_selector::WeightsLayout;
    using multi_data_tensor                 = kernel_selector::MultiDataTensor;

    using params                            = kernel_selector::Params;
    using weights_reorder_params            = kernel_selector::WeightsReorderParams;
    using generic_kernel_params             = kernel_selector::GenericKernelParams;
}

inline kernel_selector::data_type to_data_type(data_types dt)
{
    switch (dt)
    {
    case cldnn::data_types::i8:     return kernel_selector::data_type::INT8;
    case cldnn::data_types::u8:     return kernel_selector::data_type::UINT8;
    case cldnn::data_types::f16:    return kernel_selector::data_type::F16;
    case cldnn::data_types::f32:    return kernel_selector::data_type::F32;
    default:
        assert(0);
        return kernel_selector::data_type::F16;
    }
}

inline data_types from_data_type(kernel_selector::data_type dt)
{
    switch (dt)
    {
    case kernel_selector::data_type::INT8:   return cldnn::data_types::i8;
    case kernel_selector::data_type::UINT8:   return cldnn::data_types::u8;
    case kernel_selector::data_type::F16:    return cldnn::data_types::f16;
    case kernel_selector::data_type::F32:    return cldnn::data_types::f32;
    default:
        assert(0);
        return cldnn::data_types::f16;
    }
}

inline kernel_selector::weights_type to_weights_type(data_types dt)
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

inline data_types from_weights_type(kernel_selector::weights_type dt)
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

inline kernel_selector::data_layout to_data_layout(format f)
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
//     case format::brfyx:          return kernel_selector::data_layout::brfyx;
    default:
        return kernel_selector::data_layout::bfyx;
    }
}

static inline cldnn::format from_data_layout(kernel_selector::data_layout l)
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
    default:
        return cldnn::format::bfyx;
        break;
    }
}

inline kernel_selector::weights_layout to_weights_layout(format f)
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
    default:
        return kernel_selector::weights_layout::oi;
    }
}

static inline cldnn::format::type from_weights_layout(kernel_selector::weights_layout l)
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
    default:
        return cldnn::format::bfyx;
    }
}

inline kernel_selector::tuning_mode to_tuning_mode(cldnn::tuning_mode mode)
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

inline std::string to_host_version(const cldnn::version_t& version)
{
    std::stringstream ss;
    ss << version.major << "." << version.minor << "." << version.build << "." << version.revision;
    return ss.str();
}

inline kernel_selector::data_tensor convert_data_tensor(const layout& l, uint32_t split = 1, const tensor view_offset = {})
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

        offset += pitch*(add_offsets[tensor_index]);
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

inline kernel_selector::weights_tensor convert_weights_tensor(const layout& l)
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

template <typename p_type>
inline void convert_activation_func_params(const p_type primitive, kernel_selector::base_params& params)
{
    const float negative_slope = primitive->activation_negative_slope;
    if (negative_slope)
    {
        params.activationParams.m = negative_slope;
        params.activationFunc = kernel_selector::activation_function::RELU_NEGATIVE_SLOPE;
    }
    else
    {
        params.activationFunc = kernel_selector::activation_function::RELU;
    }
}

inline kernel_selector::activation_function get_kernel_selector_activation_param(cldnn_activation_func activation_func)
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
    default:
        throw std::runtime_error("Unknown activation function");
        break;
    }
}

inline kernel_selector::activation_function get_kernel_selector_activation_grad_param(cldnn_activation_grad_func activation_grad_func)
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

template <typename arg_t>
inline void convert_fused_activation_func_params(const arg_t& arg, kernel_selector::base_params& params)
{
    params.activationParams.m = arg.get_fused_activation_params().a;
    params.activationParams.n = arg.get_fused_activation_params().b;
    params.activationFunc = get_kernel_selector_activation_param(arg.get_fused_activation_func());
}

template <typename p_type>
inline void convert_new_activation_func(const p_type primitive, kernel_selector::base_params& params)
{
    params.activationFunc = get_kernel_selector_activation_param(primitive->activation_func);
    params.activationParams.m = primitive->additional_params.a;
    params.activationParams.n = primitive->additional_params.b;
}

template <typename params_t, typename arg_t>
inline params_t get_default_params(const arg_t& arg, uint32_t split = 1)
{
    params_t params;

    const auto& context = arg.get_program().get_engine().get_context();
    const auto& engine_info = context->get_engine_info();

    params.engineInfo.bSubGroupSupport      = context->extension_supported("cl_intel_subgroups");
    params.engineInfo.bSubGroupShortSupport = context->extension_supported("cl_intel_subgroups_short");
    params.engineInfo.bFP16Support          = context->extension_supported("cl_khr_fp16");
    params.engineInfo.bFP64Support          = context->extension_supported("cl_khr_fp64");
    params.engineInfo.bImageSupport         = engine_info.supports_image != 0;
    params.engineInfo.maxWorkGroupSize      = engine_info.max_work_group_size;
    params.engineInfo.maxLocalMemSize       = engine_info.max_local_mem_size;
    params.engineInfo.maxImage2dWidth       = engine_info.max_image2d_width;
    params.engineInfo.maxImage2dHeight      = engine_info.max_image2d_height;
    params.engineInfo.deviceId              = engine_info.dev_id;
    params.engineInfo.driverVersion         = engine_info.driver_version;
    params.engineInfo.hostVersion           = to_host_version(cldnn::get_version());
    
    const auto& input_layout    = arg.input().get_output_layout();
    const auto& output_layout   = arg.get_output_layout();

    params.inputs[0] = convert_data_tensor(input_layout, split);
    params.output = convert_data_tensor(output_layout, split);

    params.layerID = arg.id();

    convert_fused_activation_func_params(arg, params);

    return params;
}

template <typename params_t, typename arg_t>
inline params_t get_weights_bias_default_params(const arg_t& arg, uint32_t split = 1)
{
    params_t params = get_default_params<params_t>(arg, split);

    const auto& weights_layout = arg.weights().get_output_layout();
    params.weights = convert_weights_tensor(weights_layout);

    if (arg.bias_term())
    {
        const auto& bias_layout = arg.bias().get_output_layout();
        // bias per output is not supported on cldnn
        params.bias.push_back(convert_data_tensor(bias_layout).FlattenFeatureAndSpatials());
    }

    return params;
}

template <typename params_t, typename arg_t>
inline params_t get_default_learning_params(const arg_t& arg, uint32_t split = 1)
{
	params_t params = get_weights_bias_default_params<params_t>(arg, split);

	const auto learning_params = arg.get_program().get_options().template get<build_option_type::learning_config>()->params;

	if (arg.use_momentum())
	{
		params.use_momentum = true;
	}

	params.momentum_factor = learning_params.momentum;
	params.weights_decay = learning_params.weights_decay;

	return params;
}

template <typename optional_params_t>
inline optional_params_t get_default_optional_params(const program_impl& program)
{
    optional_params_t params;
    
    const auto& context = program.get_engine().get_context();

    params.meaningfulKernelsNames       = context->get_configuration().meaningful_kernels_names;
    params.allowStaticInputReordering   = program.get_options().get<build_option_type::optimize_data>()->enabled();
    params.allowInputReordering         = false;
    params.allowOutputReordering        = false;
    
    const auto& tuning_config = program.get_options().get<build_option_type::tuning_config>();
    params.tuningParams.mode = to_tuning_mode(tuning_config->config.mode);
    params.tuningParams.cacheFilePath = tuning_config->config.cache_file_path;

    return params;
}

template <typename optional_params_t>
inline optional_params_t get_default_weights_bias_optional_params(const program_impl& program)
{
    return get_default_optional_params<optional_params_t>(program);
}

template <typename optional_params_t>
inline optional_params_t get_default_learning_optional_params(const program_impl& program)
{
	return get_default_weights_bias_optional_params<optional_params_t>(program);
}
