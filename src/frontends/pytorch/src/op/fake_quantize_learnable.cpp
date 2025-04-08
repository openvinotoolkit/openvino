#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/add.hpp"
#include "utils.hpp"


// Helper function to create a Constant node.
std::shared_ptr<ov::Node> create_constant(const ov::element::Type& type,
    const ov::Shape& shape,
    const std::vector<float>& values) {
return ov::op::v0::Constant::create(type, shape, values);
}

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

    using namespace ov::op;

// Translate learnable fake quantize per tensor affine op.
// Expected inputs:
//   0: input tensor
//   1: scale tensor (learnable)
//   2: zero_point tensor (learnable)
//   3: quant_min (int constant)
//   4: quant_max (int constant)
//   5: grad_factor (float constant) -- (unused in forward)
// Returns: Fake quantized tensor.
OutputVector translate_fake_quantize_learnable_per_tensor_affine(const NodeContext& context) {
    // Check we have exactly 6 inputs.
    num_inputs_check(context, 6, 6);

    // Get the input nodes.
    auto input_node = context.get_input(0);
    // Convert scale and zero_point to f32 for consistency.
    auto scale_node = std::make_shared<v0::Convert>(context.get_input(1), element::f32);
    auto zero_point_node = std::make_shared<v0::Convert>(context.get_input(2), element::f32);

    auto out_low_const = context.const_input<int64_t>(3);
    auto out_high_const = context.const_input<int64_t>(4);
    // Calculate levels value - distance between bounds.
    auto levels = std::abs(out_high_const - out_low_const) + 1;
    auto out_low = std::make_shared<v0::Convert>(context.get_input(3), element::f32);
    auto out_high = std::make_shared<v0::Convert>(context.get_input(4), element::f32);

    // Normalize bounds according to quantization zero point value.
    auto out_low_normalized = std::make_shared<v1::Subtract>(out_low, zero_point_node);
    auto out_high_normalized = std::make_shared<v1::Subtract>(out_high, zero_point_node);
    // Rescale bounds according to scale value to calculate limits for input/output maximum/minimum values.
    auto bound_a = std::make_shared<v1::Multiply>(scale_node, out_low_normalized);
    auto bound_b = std::make_shared<v1::Multiply>(scale_node, out_high_normalized);
    // In case of negative scale bounds may be inverted, select maximum bound as high and minimal bound as low.
    auto bound_high = std::make_shared<v1::Maximum>(bound_a, bound_b);
    auto bound_low = std::make_shared<v1::Minimum>(bound_a, bound_b);
    return {context.mark_node(
        std::make_shared<v0::FakeQuantize>(input_node, bound_low, bound_high, bound_low, bound_high, levels))};
}

// This function translates the backward pass for the learnable per-tensor affine fake-quantize op.
// Expected inputs (in order):
//   0: dY (gradient flowing from subsequent layers)
//   1: X (original input tensor)
//   2: scale (learnable scale, a scalar tensor)
//   3: zero_point (learnable zero-point, a scalar tensor)
//   4: quant_min (int constant)
//   5: quant_max (int constant)
//   6: grad_factor (double constant used to scale gradients)
OutputVector translate_fake_quantize_learnable_per_tensor_affine_backward(const NodeContext& context) {
    // Check for exactly 7 inputs.
    num_inputs_check(context, 7, 7);
    auto dY = context.get_input(0);
    auto X = context.get_input(1);
    auto scale = context.get_input(2);
    auto zero_point = context.get_input(3);
    int64_t quant_min = context.const_input<int64_t>(4);
    int64_t quant_max = context.const_input<int64_t>(5);
    double grad_factor = context.const_input<double>(6);

    // Convert scale and zero_point to float32.
    auto scale_f32 = std::make_shared<ov::op::v0::Convert>(scale, ov::element::f32);
    auto zero_point_f32 = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f32);

    // Compute inverse scale: inv_scale = 1 / scale.
    auto one_const = create_constant(ov::element::f32, {}, {1.0f});
    auto inv_scale = std::make_shared<ov::op::v1::Divide>(one_const, scale_f32);

    // For backward, adjust zero_point (as in PyTorch, add 0.5) and clamp it.
    auto half_const = create_constant(ov::element::f32, {}, {0.5f});
    auto zero_point_adjusted = std::make_shared<ov::op::v1::Add>(zero_point_f32, half_const);
    auto quant_min_const = create_constant(ov::element::f32, {}, {static_cast<float>(quant_min)});
    auto quant_max_const = create_constant(ov::element::f32, {}, {static_cast<float>(quant_max)});
    auto clamped_zero_point = std::make_shared<ov::op::v1::Minimum>(
        std::make_shared<ov::op::v1::Maximum>(zero_point_adjusted, quant_min_const),
        quant_max_const);

    // Compute quantized tensor X_q:
    // X_q = clamp(round(X / scale + zero_point), quant_min, quant_max)
    auto X_div_scale = std::make_shared<ov::op::v1::Divide>(X, scale_f32);
    auto X_shifted = std::make_shared<ov::op::v1::Add>(X_div_scale, zero_point_f32);
    auto X_rounded = std::make_shared<ov::op::v5::Round>(X_shifted->output(0), v5::Round::RoundMode::HALF_TO_EVEN);
    auto X_clamped = std::make_shared<ov::op::v1::Minimum>(
        std::make_shared<ov::op::v1::Maximum>(X_rounded, quant_min_const),
        quant_max_const);

    // Compute fake-quantized tensor X_fq = (X_q - zero_point) * scale.
    auto X_q_minus_zp = std::make_shared<ov::op::v1::Subtract>(X_clamped, zero_point_f32);
    auto X_fq = std::make_shared<ov::op::v1::Multiply>(X_q_minus_zp, scale_f32);

    // --- Compute dX ---
    // Mask: 1 if element is not clamped (i.e. X_q is strictly between quant_min and quant_max), else 0.
    auto eq_min = std::make_shared<ov::op::v1::Equal>(X_clamped, quant_min_const);
    auto eq_max = std::make_shared<ov::op::v1::Equal>(X_clamped, quant_max_const);
    auto clamped_mask_bool = std::make_shared<ov::op::v1::LogicalOr>(eq_min, eq_max);
    auto mask_bool = std::make_shared<ov::op::v1::LogicalNot>(clamped_mask_bool);
    auto mask = std::make_shared<ov::op::v0::Convert>(mask_bool, ov::element::f32);
    auto dX = std::make_shared<ov::op::v1::Multiply>(dY, mask);

    // --- Compute dScale ---
    // Compute interior gradient: interior_grad = (X_fq - X) / scale.
    auto diff = std::make_shared<ov::op::v1::Subtract>(X_fq, X);
    auto interior_grad = std::make_shared<ov::op::v1::Divide>(diff, scale_f32);

    // For clamped elements, set:
    //   value_min = quant_min - zero_point
    //   value_max = quant_max - zero_point.
    auto value_min = std::make_shared<ov::op::v1::Subtract>(quant_min_const, zero_point_f32);
    auto value_max = std::make_shared<ov::op::v1::Subtract>(quant_max_const, zero_point_f32);

    // Use nested Select ops:
    // if (X_q == quant_min) -> value_min
    // else if (X_q == quant_max) -> value_max
    // else -> interior_grad.
    auto dScale_elem = std::make_shared<ov::op::v1::Select>(
        eq_min,
        value_min,
        std::make_shared<ov::op::v1::Select>(
            eq_max,
            value_max,
            interior_grad));

    // --- Compute dZeroPoint ---
    // For clamped elements, gradient is -scale; else 0.
    auto neg_scale = std::make_shared<ov::op::v1::Multiply>(
        create_constant(ov::element::f32, {}, {-1.0f}),
        scale_f32);
    auto dZeroPoint_elem = std::make_shared<ov::op::v1::Select>(
        clamped_mask_bool,
        neg_scale,
        create_constant(ov::element::f32, {}, {0.0f}));

    // --- Reduce dScale_elem and dZeroPoint_elem over all axes to get scalar gradients.
    // Create a constant for the reduction axes (all dimensions of X).
    auto input_shape = X.get_node_shared_ptr()->get_shape();
    std::vector<int64_t> axes_vec(input_shape.size());
    for (size_t i = 0; i < input_shape.size(); i++) {
        axes_vec[i] = static_cast<int64_t>(i);
    }
    auto axes = ov::op::v0::Constant::create(ov::element::i64, {axes_vec.size()}, axes_vec);

    auto dScale_reduced = std::make_shared<ov::op::v1::ReduceSum>(dScale_elem, axes, /*keep_dims=*/true);
    auto dZeroPoint_reduced = std::make_shared<ov::op::v1::ReduceSum>(dZeroPoint_elem, axes, /*keep_dims=*/true);

    // Multiply by grad_factor.
    auto grad_factor_const = create_constant(ov::element::f32, {}, {static_cast<float>(grad_factor)});
    auto dScale_final = std::make_shared<ov::op::v1::Multiply>(dScale_reduced, grad_factor_const);
    auto dZeroPoint_final = std::make_shared<ov::op::v1::Multiply>(dZeroPoint_reduced, grad_factor_const);

    // Return the gradients: dX, dScale, dZeroPoint.
    return {context.mark_node(dX),
            context.mark_node(dScale_final),
            context.mark_node(dZeroPoint_final)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
