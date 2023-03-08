// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/graph/fused_primitive_desc.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "intel_gpu/runtime/tensor.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/quantize.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/primitive.hpp"

#include "kernel_selector_params.h"
#include "kernel_selector_common.h"
#include "tensor_type.h"

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

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

namespace cldnn {
enum class data_types : size_t;
struct format;
struct layout;
struct program;
struct fused_primitive_desc;

kernel_selector::data_type to_data_type(data_types dt);
data_types from_data_type(kernel_selector::data_type dt);
kernel_selector::weights_type to_weights_type(data_types dt);
data_types from_weights_type(kernel_selector::weights_type dt);
kernel_selector::data_layout to_data_layout(format f);
cldnn::format from_data_layout(kernel_selector::data_layout l);
kernel_selector::weights_layout to_weights_layout(format f, bool is_grouped);
cldnn::format::type from_weights_layout(kernel_selector::weights_layout l);
kernel_selector::data_tensor convert_data_tensor(const layout& l, const tensor view_offset = tensor {});
kernel_selector::weights_tensor convert_weights_tensor(const layout& l, bool is_grouped = false);
layout from_weights_tensor(const kernel_selector::weights_tensor& t);
kernel_selector::activation_function get_kernel_selector_activation_param(activation_func activation_func);

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

std::shared_ptr<kernel_selector::fuse_params> convert_fuse_params(std::shared_ptr<NodeFuseParams> p);
void convert_fused_ops_to_legacy_activations(const kernel_impl_params& param_info, std::vector<kernel_selector::base_activation_params>& activations);
bool use_legacy_fused_ops(const kernel_impl_params& param_info);

void set_params(const kernel_impl_params& param_info, kernel_selector::params& params);

template <typename params_t>
inline params_t get_default_params(const kernel_impl_params& param_info, bool is_shape_agnostic = false) {
    params_t params;

    set_params(param_info, params);

    const auto& input_layout = param_info.get_input_layout(0);
    const auto& output_layout = param_info.get_output_layout(0);

    params.is_shape_agnostic = is_shape_agnostic;
    params.inputs[0] = convert_data_tensor(input_layout);
    params.outputs[0] = convert_data_tensor(output_layout);
    params.layerID = param_info.desc->id;

    if (use_legacy_fused_ops(param_info)) {
        // Single activation is converted to legacy fused ops format to keep good performance
        // TODO: Remove it once all kernels supports new fused ops mechanism
        convert_fused_ops_to_legacy_activations(param_info, params.activations);
    } else {
        std::map<primitive_id, std::pair<size_t, kernel_selector::Datatype>> prim_id_type_map;
        size_t op_id = 0;
        for (auto& fused_prim : param_info.fused_desc) {
            kernel_selector::fused_operation_desc desc;
            desc.op_params = convert_fuse_params(fused_prim.f_param);

            OPENVINO_ASSERT(desc.op_params != nullptr, "[GPU] Invalid fused operation (", param_info.desc->id , ") of type ", param_info.desc->type_string());


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
    }

    return params;
}

template <typename params_t>
inline params_t get_weights_bias_default_params(const kernel_impl_params& param_info, bool has_group_dimension = false, bool is_shape_agnostic = false) {
    params_t params = get_default_params<params_t>(param_info, is_shape_agnostic);
    params.weights = convert_weights_tensor(*param_info.weights_layout, has_group_dimension);

    if (param_info.bias_layout) {
        auto bias_layout = *param_info.bias_layout;
        params.bias.push_back(convert_data_tensor(bias_layout).FlattenFeatureAndSpatials());
    }

    return params;
}

template <typename params_t>
params_t get_weight_bias_zero_point_default_params(const kernel_impl_params& param_info, bool has_group_dimension = false, bool is_shape_agnostic = false) {
    params_t params = get_weights_bias_default_params<params_t>(param_info, has_group_dimension, is_shape_agnostic);

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
}  // namespace cldnn
