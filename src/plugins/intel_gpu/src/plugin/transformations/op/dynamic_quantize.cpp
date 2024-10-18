// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/dynamic_quantize.hpp"

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

DynamicQuantize::DynamicQuantize(const Output<Node>& data,
                                 const QuantizationConfig& config,
                                 const std::vector<uint64_t>& scales_zp_output_order,
                                 const bool combine_scales_and_zp)
    : ov::op::internal::DynamicQuantize(data, config, combine_scales_and_zp || config.mode == QuantizationConfig::QuantizationMode::Symmetric ? 2 : 3),
      m_combine_scales_and_zp(combine_scales_and_zp),
      m_scales_zp_output_order(scales_zp_output_order) {
    if (m_scales_zp_output_order.empty()) {
        m_scales_zp_output_order.resize(data.get_partial_shape().size());
        std::iota(m_scales_zp_output_order.begin(), m_scales_zp_output_order.end(), 0);
    }

    OPENVINO_ASSERT(data.get_partial_shape().size() == m_scales_zp_output_order.size());
    validate_and_infer_types();
}

void DynamicQuantize::validate_and_infer_types() {
    std::vector<ov::PartialShape> input_shapes = {get_input_partial_shape(0)};

    auto out_shapes = shape_infer(this, input_shapes, m_config, m_scales_zp_output_order, m_combine_scales_and_zp);
    set_output_type(0, m_config.quantization_dt, out_shapes[0]);
    set_output_type(1, m_config.scale_dt, out_shapes[1]);

    if (m_config.is_asymmetric_quantization() && !m_combine_scales_and_zp)
        set_output_type(2, m_config.zp_dt, out_shapes[2]);
}

std::shared_ptr<Node> DynamicQuantize::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<DynamicQuantize>(new_args.at(0), m_config, m_scales_zp_output_order, m_combine_scales_and_zp);
}

std::vector<ov::PartialShape> DynamicQuantize::shape_infer(const DynamicQuantize* op,
                                                           const std::vector<ov::PartialShape>& input_shapes,
                                                           const QuantizationConfig& config,
                                                           const std::vector<uint64_t>& scales_zp_output_order,
                                                           const bool combine_scales_and_zp) {
    std::vector<ov::PartialShape> out_shapes = ov::op::internal::DynamicQuantize::shape_infer(op, input_shapes, config);
    const auto is_asymmetric = config.is_asymmetric_quantization();
    if (is_asymmetric && combine_scales_and_zp) {
        out_shapes.pop_back(); // drop zero_points shape
    }

    auto transpose_shape = [](const ov::PartialShape& shape, const std::vector<uint64_t>& scales_zp_output_order) {
        auto transposed_shape = shape;
        for (size_t i = 0; i < scales_zp_output_order.size(); i++) {
            OPENVINO_ASSERT(scales_zp_output_order[i] < transposed_shape.size());
            transposed_shape[i] = shape[scales_zp_output_order[i]];
        }

        return transposed_shape;
    };

    // Transpose scales and zero points
    for (size_t i = 1; i < out_shapes.size(); i++) {
        out_shapes[i] = transpose_shape(out_shapes[i], scales_zp_output_order);
    }

    if (is_asymmetric && combine_scales_and_zp) {
        // TODO: currently scales and zero points are supposed to be combined over the last dimension only
        const auto combine_axis = out_shapes[1].size() - 1;
        OPENVINO_ASSERT(config.group_sizes[scales_zp_output_order[combine_axis]] != 1);

        out_shapes[1][combine_axis] *= 2; // (scale, zero_point) pairs
    }

    return out_shapes;
}

}  // namespace internal
}  // namespace op
}  // namespace ov
