// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/graph/fused_primitive_desc.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "intel_gpu/runtime/tensor.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/quantize.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/primitive.hpp"

#include "kernel_selector_params.h"
#include "weight_bias_params.h"
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

}  // namespace kernel_selector
namespace ov {
namespace element {
enum class Type_t;
}  // namespaec element
}  // namespaec ov
namespace cldnn {
struct format;
struct layout;
struct program;
struct fused_primitive_desc;

kernel_selector::data_type to_data_type(ov::element::Type_t dt);
ov::element::Type_t from_data_type(kernel_selector::data_type dt);
kernel_selector::weights_type to_weights_type(ov::element::Type_t dt);
ov::element::Type_t from_weights_type(kernel_selector::weights_type dt);
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

inline kernel_selector::eltwise_mode convert_to_eltwise_mode(eltwise_mode mode) {
    switch (mode) {
        case eltwise_mode::sum:
            return kernel_selector::eltwise_mode::ADD;
        case eltwise_mode::sub:
            return kernel_selector::eltwise_mode::SUB;
        case eltwise_mode::max:
            return kernel_selector::eltwise_mode::MAX;
        case eltwise_mode::prod:
            return kernel_selector::eltwise_mode::MUL;
        case eltwise_mode::div:
            return kernel_selector::eltwise_mode::DIV;
        case eltwise_mode::min:
            return kernel_selector::eltwise_mode::MIN;
        case eltwise_mode::pow:
            return kernel_selector::eltwise_mode::POW;
        case eltwise_mode::mod:
            return kernel_selector::eltwise_mode::MODULU;
        case eltwise_mode::eq:
            return kernel_selector::eltwise_mode::EQ;
        case eltwise_mode::ne:
            return kernel_selector::eltwise_mode::NE;
        case eltwise_mode::lt:
            return kernel_selector::eltwise_mode::LT;
        case eltwise_mode::le:
            return kernel_selector::eltwise_mode::LE;
        case eltwise_mode::gt:
            return kernel_selector::eltwise_mode::GT;
        case eltwise_mode::ge:
            return kernel_selector::eltwise_mode::GE;
        case eltwise_mode::logic_and:
            return kernel_selector::eltwise_mode::LOGIC_AND;
        case eltwise_mode::logic_or:
            return kernel_selector::eltwise_mode::LOGIC_OR;
        case eltwise_mode::logic_xor:
            return kernel_selector::eltwise_mode::LOGIC_XOR;
        case eltwise_mode::squared_diff:
            return kernel_selector::eltwise_mode::SQUARED_DIFF;
        case eltwise_mode::floor_mod:
            return kernel_selector::eltwise_mode::FLOOR_MOD;
        case eltwise_mode::is_finite:
            return kernel_selector::eltwise_mode::IS_FINITE;
        case eltwise_mode::is_inf:
            return kernel_selector::eltwise_mode::IS_INF;
        case eltwise_mode::is_nan:
            return kernel_selector::eltwise_mode::IS_NAN;
        case eltwise_mode::right_shift:
            return kernel_selector::eltwise_mode::RIGHT_SHIFT;
        case eltwise_mode::left_shift:
            return kernel_selector::eltwise_mode::LEFT_SHIFT;
        case eltwise_mode::bitwise_and:
            return kernel_selector::eltwise_mode::BITWISE_AND;
        case eltwise_mode::bitwise_or:
            return kernel_selector::eltwise_mode::BITWISE_OR;
        case eltwise_mode::bitwise_xor:
            return kernel_selector::eltwise_mode::BITWISE_XOR;
        default:
            OPENVINO_ASSERT(false, "Unsupported eltwise mode!");
            return kernel_selector::eltwise_mode::ADD;
    }
}

inline ov::PartialShape extend_shape_to_rank_from_end(ov::PartialShape pshape, size_t rank = 4) {
    if (pshape.size() >= rank) {
        return pshape;
    }
    pshape.insert(pshape.end(), rank - pshape.size(), ov::Dimension(1));
    return pshape;
}

inline ov::PartialShape extend_shape_to_rank_from_begin(const ov::PartialShape& pshape, size_t rank = 4) {
    if (pshape.size() >= rank) {
        return pshape;
    }
    ov::PartialShape extended_pshape(std::vector<int64_t>(rank - pshape.size(), 1));
    extended_pshape.insert(extended_pshape.end(), pshape.begin(), pshape.end());
    return extended_pshape;
}

inline bool broadcastable(const ov::PartialShape& first_pshape, const ov::PartialShape& second_pshape, bool use_new_shape_infer,
                          bool first_to_second_only = false) {
    if (first_pshape.is_dynamic() || second_pshape.is_dynamic()) {
        return false;
    }
    if (first_to_second_only) {
        if (first_pshape.size() > second_pshape.size()) {
            return false;
        }
    } else {
        if (first_pshape.size() != second_pshape.size() && use_new_shape_infer) {
            return false;
        }
    }
    size_t min_size = std::min(first_pshape.size(), second_pshape.size());

    for (size_t i = 0; i < min_size; ++i) {
        if (!(first_pshape[i] == 1 || (!first_to_second_only && second_pshape[i] == 1) || first_pshape[i] == second_pshape[i])) {
            return false;
        }
    }
    return true;
}

inline kernel_impl_params canonicalize_fused_shapes(const kernel_impl_params& impl_params) {
    auto updated_impl_params = impl_params;
    bool use_new_shape_infer = impl_params.prog->is_new_shape_infer();

    for (auto& fd : updated_impl_params.fused_desc) {
        if (fd.is_type<eltwise>() && fd.total_num_deps == 2 && fd.has_outer_dep()) {
            if (updated_impl_params.input_layouts.size() > size_t(fd.outer_dep_start_idx)) {
                const auto& out_pshape = updated_impl_params.output_layouts[0].get_partial_shape();

                auto& dep_layout = updated_impl_params.input_layouts[fd.outer_dep_start_idx];
                const auto& dep_shape = dep_layout.get_partial_shape();

                if (!broadcastable(dep_shape, out_pshape, use_new_shape_infer)) {
                    dep_layout.set_partial_shape(extend_shape_to_rank_from_begin(dep_shape, out_pshape.size()));
                }
            }
        }
    }
    return updated_impl_params;
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
        for (size_t i = fd.dep_idx_start; i < fd.dep_idx_start + fd.dep_size; i++) {
            fd.tensors.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }
    }
}

bool query_microkernels_supported(cldnn::engine& e, const cldnn::ExecutionConfig& config);

}  // namespace cldnn
