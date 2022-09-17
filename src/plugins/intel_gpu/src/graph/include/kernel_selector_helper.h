// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "intel_gpu/runtime/tensor.hpp"
#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/quantize.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/primitive.hpp"

#include "kernel_selector_params.h"
#include "kernel_selector_common.h"
#include "tensor_type.h"
#include "fused_primitive_desc.h"

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

using namespace cldnn;

namespace cldnn {
enum class data_types : size_t;
enum class tuning_mode;
struct format;
struct layout;
struct program;
struct fused_primitive_desc;
}  // namespace cldnn

namespace kernel_selector {
using n_dims = kernel_selector::Tensor::NDims;
using kernel_data = kernel_selector::KernelData;
using kernel_string = kernel_selector::KernelString;
using cl_kernel_data = kernel_selector::clKernelData;
using kernel_arguments = kernel_selector::Arguments;
using kernel_argument_element = kernel_selector::ArgumentDescriptor;
using kernel_argument_types = kernel_selector::ArgumentDescriptor::Types;
using kernel_scalar_arguments = kernel_selector::Scalars;
using kernel_scalar_argument_types = kernel_selector::ScalarDescriptor::Types;

using data_type = kernel_selector::Datatype;
using weights_type = kernel_selector::WeightsType;
using activation_function = kernel_selector::ActivationFunction;
using pool_type = kernel_selector::PoolType;
using pool_remainder = kernel_selector::PoolRemainder;
using argm_axis = kernel_selector::ArgMaxMinAxis;
using argm_output = kernel_selector::ArgMaxMinOut;
using argm_sort = kernel_selector::ArgMaxMinSortType;
using lrn_mode = kernel_selector::LRNMode;
using normalize_mode = kernel_selector::NormalizeMode;
using mvn_mode = kernel_selector::MVNMode;
using mvn_eps_mode = kernel_selector::MVNEpsMode;
using kernel_divider_mode = kernel_selector::KernelDividerMode;
using eltwise_mode = kernel_selector::EltwiseMode;
using eltwise_input_mode = kernel_selector::EltwiseInputMode;
using softmax_dim = kernel_selector::SoftmaxDim;
using mean_subtruct_mode = kernel_selector::MeanSubtractMode;
using mean_op = kernel_selector::MeanOp;
using concat_axis = kernel_selector::ConcatAxis;
using tuning_mode = kernel_selector::TuningMode;
using sample_type = kernel_selector::ResampleType;
using coordinate_transformation_mode = kernel_selector::CoordinateTransformationMode;
using nearest_mode = kernel_selector::NearestMode;
using shape_calculation_mode = kernel_selector::ShapeCalculationMode;
using interpolate_axis = kernel_selector::InterpolateAxis;
using border_type = kernel_selector::BorderType;
using gather_axis = kernel_selector::GatherAxis;
using gather_elements_axis = kernel_selector::GatherAxis;
using scatter_update_axis = kernel_selector::ScatterUpdateAxis;
using reduce_mode = kernel_selector::ReduceMode;
using cum_sum_axis = kernel_selector::CumSumAxis;
using depth_to_space_mode = kernel_selector::DepthToSpaceMode;

using data_tensor = kernel_selector::DataTensor;
using weights_tensor = kernel_selector::WeightsTensor;
template <typename T>
using dim_tensor = kernel_selector::DimTensor<T>;
using data_layout = kernel_selector::DataLayout;
using weights_layout = kernel_selector::WeightsLayout;
using multi_data_tensor = kernel_selector::MultiDataTensor;

using params = kernel_selector::Params;
using weights_reorder_params = kernel_selector::WeightsReorderParams;
using generic_kernel_params = kernel_selector::GenericKernelParams;

}  // namespace kernel_selector

kernel_selector::data_type to_data_type(data_types dt);
data_types from_data_type(kernel_selector::data_type dt);
kernel_selector::weights_type to_weights_type(data_types dt);
data_types from_weights_type(kernel_selector::weights_type dt);
kernel_selector::data_layout to_data_layout(format f);
cldnn::format from_data_layout(kernel_selector::data_layout l);
kernel_selector::weights_layout to_weights_layout(format f, bool is_grouped);
cldnn::format::type from_weights_layout(kernel_selector::weights_layout l);
kernel_selector::tuning_mode to_tuning_mode(cldnn::tuning_mode mode);
kernel_selector::data_tensor convert_data_tensor(const layout& l, uint32_t split = 1, const tensor view_offset = tensor {});
kernel_selector::weights_tensor convert_weights_tensor(const layout& l, bool is_grouped = false);
layout from_weights_tensor(const kernel_selector::weights_tensor& t);
kernel_selector::activation_function get_kernel_selector_activation_param(activation_func activation_func);

struct kernel_impl_params {
    bool has_runtime_layouts = false;
    const program& prog;
    std::shared_ptr<const primitive> desc;
    size_t unique_id;
    std::vector<layout> input_layouts;
    layout output_layout;
    std::vector<tensor> input_offsets;
    std::vector<cldnn::fused_primitive_desc> fused_desc;
    std::vector<activation_func> fused_act_funcs;
    std::vector<activation_additional_params> activation_params;

    optional_layout weights_layout = optional_layout();

    optional_layout bias_layout = optional_layout();
    optional_layout weights_zero_points_layout = optional_layout();
    optional_layout activations_zero_points_layout = optional_layout();
    optional_layout compensation_layout = optional_layout();

    std::map<size_t, memory::ptr> memory_deps = {};
    size_t primary_input_idx = 0;

    memory::ptr reordered_weights = nullptr;

    kernel_impl_params(program& _prog,
                       std::shared_ptr<const primitive> _desc,
                       size_t _uid,
                       const std::vector<layout>& _in_layouts,
                       layout _out_layout,
                       const std::vector<cldnn::fused_primitive_desc>& _fused_descs,
                       const std::vector<activation_func>& _fused_act_funcs,
                       const std::vector<activation_additional_params>& _act_params)
                       : has_runtime_layouts(true)
                       , prog(_prog)
                       , desc(_desc)
                       , unique_id(_uid)
                       , input_layouts(_in_layouts)
                       , output_layout(_out_layout)
                       , fused_desc(_fused_descs)
                       , fused_act_funcs(_fused_act_funcs)
                       , activation_params(_act_params)
                       , primary_input_idx(0) {
    }

    layout get_input_layout(size_t idx = 0) const {
        OPENVINO_ASSERT(input_layouts.size() > idx,
                        "The size of input layouts must be greater than the requested index: ",
                        "Requested index is ", idx, ", ",
                        "but the size of input layouts is ", input_layouts.size());
        return input_layouts[idx];
    }

    layout get_non_padded_input_layout(size_t idx = 0) const {
        auto input_layout = get_input_layout(idx);
        auto result = layout({input_layout.get_partial_shape(), input_layout.data_type, input_layout.format});
        return result;
    }

    bool has_fused_primitives() const { return !fused_desc.empty(); }

    layout get_fused_output_layout() const {
        if (fused_desc.empty())
            return layout(data_types::f32, format::bfyx, tensor());
        return fused_desc.back().output_layout;
    }

    template <class PType>
    std::shared_ptr<const PType> typed_desc() const { return std::static_pointer_cast<const PType>(desc); }
};

template <typename T = std::uint32_t>
kernel_selector::dim_tensor<T> convert_dim_vector(const tensor& t) {
    const auto& sizes = t.sizes(format::bfwzyx);
    return {static_cast<T>(sizes[0]),
            static_cast<T>(sizes[1]),
            static_cast<T>(sizes[2]),
            static_cast<T>(sizes[3]),
            static_cast<T>(sizes[4]),
            static_cast<T>(sizes[5])};
}

template <typename p_type>
inline void convert_activation_func_params(const p_type primitive, std::vector<kernel_selector::base_activation_params>& params) {
    const float negative_slope = primitive->activation_negative_slope;
    if (negative_slope != 0.0f) {
        params.emplace_back(kernel_selector::activation_function::RELU_NEGATIVE_SLOPE, negative_slope, 0.0f);
    } else {
        params.emplace_back(kernel_selector::activation_function::RELU, 0.0f, 0.0f);
    }
}

inline void convert_fused_activation_func_params(const kernel_impl_params& param_info, std::vector<kernel_selector::base_activation_params>& params) {
    const auto& act_funcs = param_info.fused_act_funcs;
    const auto& act_params = param_info.activation_params;
    for (size_t i = 0; i < act_funcs.size(); i++) {
        params.emplace_back(get_kernel_selector_activation_param(act_funcs[i]),
                            act_params[i].a,
                            act_params[i].b);
    }
}
template <typename p_type>
inline void convert_new_activation_func(const p_type primitive, std::vector<kernel_selector::base_activation_params>& params) {
    params.insert(params.begin(), {get_kernel_selector_activation_param(primitive->activation_function),
                                   primitive->additional_params.a,
                                   primitive->additional_params.b});
}

void set_params(const kernel_impl_params& param_info, kernel_selector::params& params);

template <typename params_t>
inline params_t get_default_params(const kernel_impl_params& param_info, uint32_t split = 1) {
    params_t params;

    set_params(param_info, params);

    const auto& input_layout = param_info.get_input_layout(0);
    const auto& output_layout = param_info.output_layout;

    params.inputs[0] = convert_data_tensor(input_layout, split);
    params.outputs[0] = convert_data_tensor(output_layout, split);
    params.layerID = param_info.desc->id;

    convert_fused_activation_func_params(param_info, params.activations);
    std::map<primitive_id, std::pair<size_t, kernel_selector::Datatype>> prim_id_type_map;
    size_t op_id = 0;
    for (auto& fused_prim : param_info.fused_desc) {
        kernel_selector::fused_operation_desc desc;
        desc.op_params = std::move(fused_prim.f_param);

        if (!desc.op_params) {
            CLDNN_ERROR_MESSAGE(param_info.desc->id, "Invalid fused operation (" + param_info.desc->id + ") of type " +
                                           param_info.desc->type_string());
        }

        desc.dep_idx_start = fused_prim.dep_start_idx;
        desc.dep_size = fused_prim.deps.size();
        desc.op_id = op_id++;
        desc.output_tensor = convert_data_tensor(fused_prim.output_layout);
        prim_id_type_map[fused_prim.desc->id] = std::make_pair(desc.op_id, desc.output_tensor.GetDType());

        for (size_t i = desc.dep_idx_start; i < desc.dep_idx_start + desc.dep_size; i++) {
            desc.tensors.push_back(convert_data_tensor(param_info.get_input_layout(i)));
        }

        if (fused_prim.total_num_deps > 0) {
            desc.dep_data.resize(fused_prim.total_num_deps);
            for (auto& dep : fused_prim.fused_deps) {
                auto iter = prim_id_type_map.find(dep.first);
                if (iter != prim_id_type_map.end()) {
                    auto& op_data = iter->second;
                    desc.dep_data[dep.second].dep_type  = kernel_selector::DepType::INTERNAL;
                    desc.dep_data[dep.second].op_id     = op_data.first;
                    desc.dep_data[dep.second].data_type = op_data.second;
                }
            }

            int idx = 0;
            for (auto& dep : fused_prim.deps) {
                desc.dep_data[dep.second].dep_type  = kernel_selector::DepType::EXTERNAL;
                desc.dep_data[dep.second].op_id     = idx;
                desc.dep_data[dep.second].data_type = desc.tensors[idx++].GetDType();
            }

            for (auto& dep : desc.dep_data) {
                if (dep.dep_type == kernel_selector::DepType::UNDEFINED) {
                    dep.dep_type    = kernel_selector::DepType::ORIGINAL;
                    break;
                }
            }
        }
        params.fused_ops.push_back(desc);
    }
    return params;
}

template <typename params_t>
inline params_t get_weights_bias_default_params(const kernel_impl_params& param_info, uint32_t split = 1, uint32_t groups = 1,
                                                bool has_group_dimension = false) {
    params_t params = get_default_params<params_t>(param_info, split);
    params.weights = convert_weights_tensor(*param_info.weights_layout, has_group_dimension);

    if (param_info.bias_layout) {
        auto bias_layout = *param_info.bias_layout;
        if (groups != 1) {
            auto bias_size = bias_layout.get_tensor();
            bias_size.feature[0] /= static_cast<int>(groups);
            bias_layout.set_tensor(bias_size);
        }
        params.bias.push_back(convert_data_tensor(bias_layout).FlattenFeatureAndSpatials());
    }

    return params;
}

template <typename params_t>
params_t get_weight_bias_zero_point_default_params(const kernel_impl_params& param_info, uint32_t split = 1, uint32_t groups = 1,
                                                   bool has_group_dimension = false) {
    params_t params = get_weights_bias_default_params<params_t>(param_info, split, groups, has_group_dimension);

    if (param_info.weights_zero_points_layout) {
        params.weights_zero_points.push_back(
            convert_data_tensor(*param_info.weights_zero_points_layout)
            .FlattenFeatureAndSpatials());
    }

    if (param_info.activations_zero_points_layout) {
        params.activations_zero_points.push_back(
            convert_data_tensor(*param_info.activations_zero_points_layout)
            .FlattenFeatureAndSpatials());
    }

    if (param_info.compensation_layout) {
        params.compensation.push_back(
            convert_data_tensor(*param_info.compensation_layout).FlattenFeatureAndSpatials());
    }

    return params;
}

void set_optional_params(const program& program, kernel_selector::optional_params& params);

template <typename optional_params_t>
inline optional_params_t get_default_optional_params(const program& program) {
    optional_params_t params;
    set_optional_params(program, params);
    return params;
}

template <typename optional_params_t>
inline optional_params_t get_default_weights_bias_optional_params(const program& program) {
    return get_default_optional_params<optional_params_t>(program);
}
