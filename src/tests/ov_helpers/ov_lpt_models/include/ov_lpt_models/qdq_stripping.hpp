// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "openvino/core/model.hpp"

namespace ov {
namespace builder {
namespace subgraph {

struct QuantizationParams {
    float i_l;
    float i_h;
    float o_l;
    float o_h;
    int zero_point;
};

class QDQStrippingFunction {
public:
    static ov::Output<ov::Node> build_fq(const ov::Output<ov::Node>& input,
                                         const QuantizationParams& qp,
                                         size_t levels = 65536);
    static ov::Output<ov::Node> build_dq(const ov::Output<ov::Node>& input,
                                         const ov::element::Type& quantization_precision,
                                         const QuantizationParams& qp,
                                         bool convert_on_zero_point = true);

    // Builds a weight-DQ pattern: Constant(quantized_type) -> Convert(f32) -> Subtract(zp) -> Multiply(scale)
    // When seed is provided, generates random constant values using make_constant.
    // When constant_values vector is provided, uses those values directly.
    // Otherwise uses single constant_value
    static ov::Output<ov::Node> build_weights_dq(ov::element::Type quantized_type,
                                                 const ov::Shape& shape,
                                                 float scale_value,
                                                 int zero_point = 0,
                                                 std::optional<size_t> seed = std::nullopt,
                                                 std::optional<std::vector<int>> constant_values = std::nullopt,
                                                 float constant_value = 1.f);

    // Builds Conv bias pattern (coming from ONNX FE conversion) using ShapeOf-based reshape
    static ov::Output<ov::Node> add_bias(const ov::Output<ov::Node>& conv, const ov::Output<ov::Node>& bias);

    static std::shared_ptr<ov::Model> build_shared_dq_pattern(const ov::PartialShape& input_shape,
                                                              const ov::element::Type& quantization_precision);
    static std::shared_ptr<ov::Model> build_shared_dq_pattern_ref(const ov::PartialShape& input_shape,
                                                                  bool need_weights_adjustment = true);

    static std::shared_ptr<ov::Model> build_mul_matmul_pattern(const ov::PartialShape& input_shape,
                                                               const ov::element::Type& quantization_precision);
    static std::shared_ptr<ov::Model> build_mul_matmul_pattern_ref(const ov::PartialShape& input_shape,
                                                                   bool need_weights_adjustment = true);

    static std::shared_ptr<ov::Model> build_matmul_with_bias_pattern(const ov::PartialShape& input_shape,
                                                                     const ov::element::Type& quantization_precision);
    static std::shared_ptr<ov::Model> build_matmul_with_bias_pattern_ref(const ov::PartialShape& input_shape,
                                                                         bool need_weights_adjustment = true);

    static std::shared_ptr<ov::Model> build_residual_block_pattern(const ov::PartialShape& input_shape,
                                                                   const ov::element::Type& quantization_precision,
                                                                   bool skip_final_mvn = false);
    static std::shared_ptr<ov::Model> build_residual_block_pattern_ref(const ov::PartialShape& input_shape,
                                                                       bool need_weights_adjustment = true,
                                                                       bool skip_final_mvn = false);

    static std::shared_ptr<ov::Model> build_forward_bias_pattern(const ov::PartialShape& input_shape,
                                                                 const ov::element::Type& quantization_precision);
    static std::shared_ptr<ov::Model> build_forward_bias_pattern_ref(const ov::PartialShape& input_shape,
                                                                     bool need_weights_adjustment = true);

private:
    // Builds one residual block: MVN → Conv+bias [→ optional FQ] + shortcut → Add.
    // Shared by build_residual_block_pattern (source) and build_residual_block_pattern_ref (ref).
    static ov::Output<ov::Node> create_residual_block(
        const ov::Output<ov::Node>& input,
        size_t seed,
        float weight_scale,
        float bias_scale,
        bool add_shortcut_conv = false,
        std::optional<std::pair<float, float>> branch_fq_range = std::nullopt);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
