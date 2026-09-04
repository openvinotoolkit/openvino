// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common_utils/eltwise_kernel_params.hpp"
#include "common_utils/kernel_selector_data_adapter.hpp"
#include "common_utils/shape_utils.hpp"
#include "intel_gpu/graph/fused_primitive_desc.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/primitive.hpp"
#include "intel_gpu/primitives/quantize.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/tensor.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "kernel_selector_common.h"
#include "kernel_selector_params.h"
#include "tensor_type.h"
#include "weight_bias_params.h"

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

}  // namespace kernel_selector
namespace ov {
namespace element {
enum class Type_t;
}  // namespace element
}  // namespace ov
namespace cldnn {
struct format;
struct layout;
struct program;
struct fused_primitive_desc;

ov::element::Type_t from_data_type(kernel_selector::data_type dt);
kernel_selector::weights_type to_weights_type(ov::element::Type_t dt);
ov::element::Type_t from_weights_type(kernel_selector::weights_type dt);
cldnn::format from_data_layout(kernel_selector::data_layout l);
kernel_selector::weights_layout to_weights_layout(format f, bool is_grouped);
cldnn::format::type from_weights_layout(kernel_selector::weights_layout l);
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
void set_default_params(const kernel_impl_params& param_info, kernel_selector::base_params& params, bool is_shape_agnostic);
void set_dynamic_shape_offsets(kernel_selector::params& params);
void set_weights_bias_default_params(const kernel_impl_params& param_info,
                                     kernel_selector::weight_bias_params& params,
                                     bool has_group_dimension,
                                     bool is_shape_agnostic);
void set_weight_bias_zero_point_default_params(const kernel_impl_params& param_info,
                                               kernel_selector::weight_bias_zero_point_params& params,
                                               bool has_group_dimension,
                                               bool is_shape_agnostic);

template <typename params_t>
inline params_t get_default_params(const kernel_impl_params& param_info, bool is_shape_agnostic = false) {
    params_t params = params_t();
    set_default_params(param_info, params, is_shape_agnostic);
    return params;
}

template <typename params_t>
inline params_t get_weights_bias_default_params(const kernel_impl_params& param_info, bool has_group_dimension = false, bool is_shape_agnostic = false) {
    params_t params;
    set_weights_bias_default_params(param_info, params, has_group_dimension, is_shape_agnostic);
    return params;
}

template <typename params_t>
params_t get_weight_bias_zero_point_default_params(const kernel_impl_params& param_info, bool has_group_dimension = false, bool is_shape_agnostic = false) {
    params_t params;
    set_weight_bias_zero_point_default_params(param_info, params, has_group_dimension, is_shape_agnostic);
    return params;
}

inline bool broadcastable(const ov::PartialShape& first_pshape,
                          const ov::PartialShape& second_pshape,
                          bool use_new_shape_infer,
                          bool first_to_second_only = false) {
    return shapes_are_broadcastable(first_pshape, second_pshape, use_new_shape_infer, first_to_second_only);
}

inline std::shared_ptr<WeightsReorderParams> create_weights_reorder_params(const kernel_selector::WeightsReorderParams& params) {
    if (!params.is_initialized) {
        return nullptr;
    }

    return std::make_shared<WeightsReorderParams>(from_weights_tensor(params.src), from_weights_tensor(params.dest), params.rotate);
}

inline void update_shapes(kernel_selector::Params& p, const kernel_impl_params& impl_param) {
    auto& bp = static_cast<kernel_selector::base_params&>(p);
    for (size_t i = 0; i < bp.inputs.size(); i++) {
        bp.inputs[i] = convert_data_tensor(impl_param.input_layouts[i]);
    }
    for (size_t i = 0; i < bp.outputs.size(); i++) {
        bp.outputs[i] = convert_data_tensor(impl_param.output_layouts[i]);
    }

    for (size_t i = 0; i < bp.fused_ops.size(); i++) {
        const auto& fused_prim = impl_param.fused_desc[i];
        auto& fd = bp.fused_ops[i];
        fd.output_tensor = convert_data_tensor(fused_prim.output_layout);
        fd.tensors.clear();
        for (size_t i = fd.dep_idx_start; i < fd.dep_idx_start + fd.dep_size; i++) {
            fd.tensors.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }
    }
}

bool check_cm_jit_support(cldnn::engine& e, const cldnn::ExecutionConfig& config);
bool query_microkernels_supported(cldnn::engine& e, const cldnn::ExecutionConfig& config);
bool query_register_file_size_option_supported(cldnn::engine& e, const cldnn::ExecutionConfig& config);

}  // namespace cldnn
