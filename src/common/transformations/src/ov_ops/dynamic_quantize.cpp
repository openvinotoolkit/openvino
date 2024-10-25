// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/dynamic_quantize.hpp"

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace op {
namespace internal {

DynamicQuantize::DynamicQuantize(const Output<Node>& data,
                                 const QuantizationConfig& config,
                                 const OutputStorageType& output_storage,
                                 const std::vector<uint64_t>& scales_zp_output_order)
    : Op({data}),
      m_output_storage_type(output_storage),
      m_scales_zp_output_order(scales_zp_output_order),
      m_config(config) {
    if (m_scales_zp_output_order.empty()) {
        m_scales_zp_output_order.resize(data.get_partial_shape().size());
        std::iota(m_scales_zp_output_order.begin(), m_scales_zp_output_order.end(), 0);
    }

    OPENVINO_ASSERT(data.get_partial_shape().rank() == m_config.group_sizes.size(),
                    "DQ input rank should be same as the rank of group_size ",
                    data.get_tensor_ptr()->get_partial_shape().rank(),
                    " / ",
                    m_config.group_sizes.size());

    OPENVINO_ASSERT(data.get_partial_shape().size() == m_scales_zp_output_order.size(),
                    "DQ input rank should be same as the rank of scales and zero points output order)");

    size_t outputs_number = 2;
    if (config.is_asymmetric_quantization() && output_storage == OutputStorageType::Planar)
        outputs_number = 3;

    OPENVINO_ASSERT((output_storage == OutputStorageType::Planar) ||
                        (config.is_asymmetric_quantization() && config.scale_dt == config.zp_dt),
                    "Scales and Zero Points should have the same data type to be stored in the single buffer");

    set_output_size(outputs_number);
    validate_and_infer_types();
}

void DynamicQuantize::validate_and_infer_types() {
    std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0)};

    auto out_shapes = shape_infer(this, input_shapes, m_config, m_output_storage_type, m_scales_zp_output_order);
    set_output_type(0, m_config.quantization_dt, out_shapes[0]);
    set_output_type(1, m_config.scale_dt, out_shapes[1]);

    if (m_config.is_asymmetric_quantization() && m_output_storage_type == OutputStorageType::Planar)
        set_output_type(2, m_config.zp_dt, out_shapes[2]);
}

std::shared_ptr<Node> DynamicQuantize::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<DynamicQuantize>(new_args.at(0), m_config, m_output_storage_type, m_scales_zp_output_order);
}

std::vector<ov::PartialShape> DynamicQuantize::shape_infer(const DynamicQuantize* op,
                                                           const std::vector<ov::PartialShape>& input_shapes,
                                                           const QuantizationConfig& config,
                                                           const OutputStorageType& output_storage,
                                                           const std::vector<uint64_t>& scales_zp_output_order) {
    const auto& group_sizes = config.group_sizes;
    std::vector<ov::PartialShape> out_shapes;
    out_shapes.push_back(input_shapes[0]);

    auto scale_shape = input_shapes[0];
    OPENVINO_ASSERT(scale_shape.size() == group_sizes.size(),
                    "Scale_shape and group_size are supposed to have same rank: ",
                    scale_shape.size(),
                    " / ",
                    group_sizes.size());
    for (size_t i = 0; i < scale_shape.size(); i++) {
        if (scale_shape[i].is_dynamic() || scale_shape[i] == 0)
            continue;

        if (group_sizes[i] == UINT64_MAX) {
            scale_shape[i] = 1;
        } else {
            scale_shape[i] = ov::util::ceil_div(scale_shape[i].get_length(), static_cast<int64_t>(group_sizes[i]));
        }
    }
    out_shapes.push_back(scale_shape);

    // Add zero points shape, same as the scales
    if (config.is_asymmetric_quantization() && output_storage == OutputStorageType::Planar)
        out_shapes.push_back(scale_shape);

    auto transpose_shape = [](const ov::PartialShape& shape, const std::vector<uint64_t>& scales_zp_output_order) {
        auto transposed_shape = shape;
        for (size_t i = 0; i < scales_zp_output_order.size(); i++) {
            OPENVINO_ASSERT(scales_zp_output_order[i] < transposed_shape.size());
            transposed_shape[i] = shape[scales_zp_output_order[i]];
        }

        return transposed_shape;
    };

    // Transpose scales and zero points shapes
    for (size_t i = 1; i < out_shapes.size(); i++) {
        out_shapes[i] = transpose_shape(out_shapes[i], scales_zp_output_order);
    }

    if (config.is_asymmetric_quantization() && output_storage != OutputStorageType::Planar) {
        // Currently scales and zero points are supposed to be combined over the last dimension only
        const auto combine_axis = out_shapes[1].size() - 1;
        OPENVINO_ASSERT(config.group_sizes[scales_zp_output_order[combine_axis]] != 1);

        out_shapes[1][combine_axis] *= 2;  // [scale, zero_point] pairs
    }

    return out_shapes;
}

}  // namespace internal
}  // namespace op
}  // namespace ov
