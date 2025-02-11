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

DynamicQuantize::DynamicQuantize(const Output<Node>& data, const Attributes& attrs) : Op({data}), m_attrs(attrs) {
    if (m_attrs.scales_zp_output_order.empty()) {
        m_attrs.scales_zp_output_order.resize(data.get_partial_shape().size());
        std::iota(m_attrs.scales_zp_output_order.begin(), m_attrs.scales_zp_output_order.end(), 0);
    }

    OPENVINO_ASSERT(data.get_partial_shape().rank() == m_attrs.group_sizes.size(),
                    "DQ input rank should be same as the rank of group_size ",
                    data.get_tensor_ptr()->get_partial_shape().rank(),
                    " / ",
                    m_attrs.group_sizes.size());

    OPENVINO_ASSERT(data.get_partial_shape().size() == m_attrs.scales_zp_output_order.size(),
                    "DQ input rank should be same as the rank of scales and zero points output order)");

    size_t outputs_number = 2;
    if (m_attrs.quantization_type == QuantizationType::Asymmetric &&
        m_attrs.output_storage_type == OutputStorageType::Planar)
        outputs_number = 3;

    OPENVINO_ASSERT(
        (m_attrs.output_storage_type == OutputStorageType::Planar) ||
            (m_attrs.quantization_type == QuantizationType::Asymmetric && m_attrs.scale_dt == m_attrs.zp_dt),
        "Scales and Zero Points should have the same data type to be stored in the single buffer");

    set_output_size(outputs_number);
    validate_and_infer_types();
}

void DynamicQuantize::validate_and_infer_types() {
    std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0)};

    auto out_shapes = shape_infer(this, input_shapes);
    set_output_type(0, m_attrs.quantization_dt, out_shapes[0]);
    set_output_type(1, m_attrs.scale_dt, out_shapes[1]);

    if (m_attrs.quantization_type == QuantizationType::Asymmetric &&
        m_attrs.output_storage_type == OutputStorageType::Planar)
        set_output_type(2, m_attrs.zp_dt, out_shapes[2]);
}

std::shared_ptr<Node> DynamicQuantize::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<DynamicQuantize>(new_args.at(0), m_attrs);
}

std::vector<ov::PartialShape> DynamicQuantize::shape_infer(const DynamicQuantize* op,
                                                           const std::vector<ov::PartialShape>& input_shapes) {
    std::vector<ov::PartialShape> out_shapes;
    out_shapes.push_back(input_shapes[0]);

    auto scale_shape = input_shapes[0];
    const auto& group_sizes = op->m_attrs.group_sizes;
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
    if (op->m_attrs.quantization_type == QuantizationType::Asymmetric &&
        op->m_attrs.output_storage_type == OutputStorageType::Planar)
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
    const auto& scales_zp_output_order = op->m_attrs.scales_zp_output_order;
    for (size_t i = 1; i < out_shapes.size(); i++) {
        out_shapes[i] = transpose_shape(out_shapes[i], scales_zp_output_order);
    }

    if (op->m_attrs.quantization_type == QuantizationType::Asymmetric &&
        op->m_attrs.output_storage_type != OutputStorageType::Planar) {
        // Currently scales and zero points are supposed to be combined over the last dimension only
        const auto combine_axis = scales_zp_output_order.empty() ? out_shapes[1].size() - 1
                                                                 : scales_zp_output_order[out_shapes[1].size() - 1];
        OPENVINO_ASSERT(group_sizes[combine_axis] != 1);

        out_shapes[1][combine_axis] *= 2;  // [scale, zero_point] pairs
    }

    return out_shapes;
}

}  // namespace internal
}  // namespace op
}  // namespace ov
