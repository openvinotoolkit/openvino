// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_kernel_params.hpp"

#include <utility>

#include "common_utils/kernel_selector_data_adapter.hpp"
#include "common_utils/shape_utils.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "openvino/core/except.hpp"
#include "program_node.h"

namespace cldnn {

kernel_selector::EltwiseMode convert_to_eltwise_mode(eltwise_mode mode) {
    switch (mode) {
    case eltwise_mode::sum:
        return kernel_selector::EltwiseMode::ADD;
    case eltwise_mode::sub:
        return kernel_selector::EltwiseMode::SUB;
    case eltwise_mode::max:
        return kernel_selector::EltwiseMode::MAX;
    case eltwise_mode::prod:
        return kernel_selector::EltwiseMode::MUL;
    case eltwise_mode::div:
        return kernel_selector::EltwiseMode::DIV;
    case eltwise_mode::min:
        return kernel_selector::EltwiseMode::MIN;
    case eltwise_mode::pow:
        return kernel_selector::EltwiseMode::POW;
    case eltwise_mode::mod:
        return kernel_selector::EltwiseMode::MODULU;
    case eltwise_mode::eq:
        return kernel_selector::EltwiseMode::EQ;
    case eltwise_mode::ne:
        return kernel_selector::EltwiseMode::NE;
    case eltwise_mode::lt:
        return kernel_selector::EltwiseMode::LT;
    case eltwise_mode::le:
        return kernel_selector::EltwiseMode::LE;
    case eltwise_mode::gt:
        return kernel_selector::EltwiseMode::GT;
    case eltwise_mode::ge:
        return kernel_selector::EltwiseMode::GE;
    case eltwise_mode::logic_and:
        return kernel_selector::EltwiseMode::LOGIC_AND;
    case eltwise_mode::logic_or:
        return kernel_selector::EltwiseMode::LOGIC_OR;
    case eltwise_mode::logic_xor:
        return kernel_selector::EltwiseMode::LOGIC_XOR;
    case eltwise_mode::squared_diff:
        return kernel_selector::EltwiseMode::SQUARED_DIFF;
    case eltwise_mode::floor_mod:
        return kernel_selector::EltwiseMode::FLOOR_MOD;
    case eltwise_mode::is_finite:
        return kernel_selector::EltwiseMode::IS_FINITE;
    case eltwise_mode::is_inf:
        return kernel_selector::EltwiseMode::IS_INF;
    case eltwise_mode::is_nan:
        return kernel_selector::EltwiseMode::IS_NAN;
    case eltwise_mode::right_shift:
        return kernel_selector::EltwiseMode::RIGHT_SHIFT;
    case eltwise_mode::left_shift:
        return kernel_selector::EltwiseMode::LEFT_SHIFT;
    case eltwise_mode::bitwise_and:
        return kernel_selector::EltwiseMode::BITWISE_AND;
    case eltwise_mode::bitwise_or:
        return kernel_selector::EltwiseMode::BITWISE_OR;
    case eltwise_mode::bitwise_xor:
        return kernel_selector::EltwiseMode::BITWISE_XOR;
    case eltwise_mode::atan2:
        return kernel_selector::EltwiseMode::ATAN2;
    default:
        OPENVINO_ASSERT(false, "Unsupported eltwise mode!");
        return kernel_selector::EltwiseMode::ADD;
    }
}

kernel_selector::eltwise_params lower_eltwise_params(const kernel_impl_params& impl_params,
                                                     kernel_selector::eltwise_params params) {
    const auto& primitive = impl_params.typed_desc<eltwise>();
    const auto input_count = primitive->input.size();
    const auto mode = convert_to_eltwise_mode(primitive->mode);

    for (size_t index = 1; index < input_count; ++index) {
        params.inputs.push_back(convert_data_tensor(impl_params.input_layouts[index]));
    }

    if (input_count == 1) {
        params.operations.push_back({{kernel_selector::eltwise_params::InputType::Buffer(0)}, mode});
    } else {
        params.operations.push_back({{kernel_selector::eltwise_params::InputType::Buffer(0),
                                      kernel_selector::eltwise_params::InputType::Buffer(1)},
                                     mode});
    }

    for (uint32_t index = 2; index < static_cast<uint32_t>(input_count); ++index) {
        params.operations.push_back({{kernel_selector::eltwise_params::InputType::Intermediate(index - 2),
                                      kernel_selector::eltwise_params::InputType::Buffer(index)},
                                     mode});
    }

    params.coefficients = primitive->coefficients;
    if (impl_params.get_program().get_node(primitive->id).is_dynamic()) {
        params.broadcast = true;
    } else {
        for (size_t index = 0; index < params.inputs.size(); ++index) {
            if (params.inputs[index].SameDims(params.outputs[0])) {
                continue;
            }

            const auto input_size = impl_params.input_layouts[index].get_tensor().raw.vector();
            const auto output_size = impl_params.get_output_layout().get_tensor().raw.vector();
            bool broadcasts_dimension = false;
            for (size_t dimension = 0; dimension < output_size.size(); ++dimension) {
                if (output_size[dimension] != 1 && input_size[dimension] == 1) {
                    broadcasts_dimension = true;
                }
            }
            if (broadcasts_dimension) {
                params.broadcast = true;
                break;
            }
            params.layoutBased = true;
            break;
        }
    }

    if (!primitive->stride.empty()) {
        params.stride.resize(primitive->stride.size());
        for (size_t index = 0; index < primitive->stride.size(); ++index) {
            params.stride[index] = {static_cast<uint32_t>(primitive->stride[index].spatial[0]),
                                    static_cast<uint32_t>(primitive->stride[index].spatial[1]),
                                    static_cast<uint32_t>(primitive->stride[index].spatial[2])};
        }
    }

    if (!params.stride.empty()) {
        const auto& reference_stride = params.stride[0];
        for (size_t index = 1; index < params.stride.size(); ++index) {
            if (reference_stride.x != params.stride[index].x || reference_stride.y != params.stride[index].y) {
                params.layoutBased = true;
            }
        }
    } else if (params.inputs.size() > 1 && !params.inputs[0].SameDimsSizes(params.inputs[1])) {
        params.broadcast = true;
    }

    params.int8_quantization = true;
    for (size_t index = 0; index < input_count; ++index) {
        if (impl_params.input_layouts[index].data_type != data_types::u8 &&
            impl_params.input_layouts[index].data_type != data_types::i8) {
            params.int8_quantization = false;
        }
    }

    return params;
}

kernel_impl_params canonicalize_eltwise_shapes(const kernel_impl_params& impl_params) {
    auto canonical_params = canonicalize_fused_shapes(impl_params);
    const bool use_new_shape_infer = impl_params.prog->is_new_shape_infer();

    auto& output_layout = canonical_params.output_layouts[0];
    const auto output_shape = output_layout.get_partial_shape();
    output_layout.set_partial_shape(extend_shape_to_rank_from_end(output_shape));

    for (auto& input_layout : canonical_params.input_layouts) {
        auto input_shape = input_layout.get_partial_shape();
        if (!shapes_are_broadcastable(input_shape, output_shape, use_new_shape_infer)) {
            input_shape = extend_shape_to_rank_from_begin(input_shape, output_shape.size());
        }
        input_layout.set_partial_shape(extend_shape_to_rank_from_end(input_shape));
        input_layout.format = format::adjust_to_rank(input_layout.format, input_shape.size());
    }

    return canonical_params;
}

}  // namespace cldnn
