// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_linear(const NodeContext& context) {
    // schema: aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto weight = context.get_input(1);
    auto matmul = context.mark_node(std::make_shared<v0::MatMul>(x, weight, false, true));
    if (!context.input_is_none(2)) {
        auto bias = context.get_input(2);
        matmul = context.mark_node(std::make_shared<v1::Add>(matmul, bias));
    }
    return {matmul};
};

OutputVector translate_linear_ext(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto initial_x = x;
    auto weight = context.get_input(1);
    bool convert_back = false;
    if (weight.get_element_type() != element::f32) {
        // In case of patched linear it can have mixed fp16/bf16 and fp32 input type.
        // In other cases these conversion is not required.
        weight = context.mark_node(std::make_shared<v0::Convert>(weight, element::f32));
        if (x.get_element_type() != element::f32) {
            // Convert to f32
            x = context.mark_node(std::make_shared<v0::Convert>(x, element::f32));
            convert_back = true;
        }
    }
    auto matmul = context.mark_node(std::make_shared<v0::MatMul>(x, weight, false, true));
    if (!context.input_is_none(2)) {
        auto bias = context.get_input(2);

        if (bias.get_element_type() != element::f32) {
            // Same reason as for weight.
            bias = context.mark_node(std::make_shared<v0::Convert>(bias, element::f32));
        }
        matmul = context.mark_node(std::make_shared<v1::Add>(matmul, bias));
    }
    if (convert_back) {
        matmul = context.mark_node(std::make_shared<v1::ConvertLike>(matmul, initial_x));
    }
    return {matmul};
};

namespace {

// Write a u4 value at a given linear index in a packed u4 buffer.
inline void set_u4(uint8_t* data, size_t idx, uint8_t val) {
    size_t byte_idx = idx / 2;
    if (idx & 1)
        data[byte_idx] = (data[byte_idx] & 0x0F) | static_cast<uint8_t>((val & 0x0F) << 4);
    else
        data[byte_idx] = (data[byte_idx] & 0xF0) | (val & 0x0F);
}

Output<Node> low_precision_subgraph(const NodeContext& context,
                                    const Output<Node>& x,
                                    const Output<Node>& weights,
                                    const Output<Node>& zero_points,
                                    const Output<Node>& scales,
                                    const Output<Node>& out_shape) {
    auto new_qweight = context.mark_node(std::make_shared<v0::Convert>(weights, scales.get_element_type()));
    auto new_qzeros = context.mark_node(std::make_shared<v0::Convert>(zero_points, scales.get_element_type()));

    auto w_s = context.mark_node(std::make_shared<v1::Subtract>(new_qweight, new_qzeros));
    auto weight = context.mark_node(std::make_shared<v1::Multiply>(w_s, scales));
    auto weight_shape = weights.get_shape();
    if (out_shape.get_node() != nullptr) {
        weight = context.mark_node(std::make_shared<v1::Reshape>(weight, out_shape, false));
    }
    weight = context.mark_node(std::make_shared<v1::ConvertLike>(weight, x));
    return weight;
}

uint32_t rearrange_awq_bits(uint32_t num) {
    uint32_t result = 0;
    uint32_t mask = 0xF;

    // Rearrange each 4-bit part in accordance with the AWQ i32->u4 unpacking schema
    result |= (num & (mask << 0)) << 0;
    result |= (num & (mask << 16)) >> 12;
    result |= (num & (mask << 4)) << 4;
    result |= (num & (mask << 20)) >> 8;
    result |= (num & (mask << 8)) << 8;
    result |= (num & (mask << 24)) >> 4;
    result |= (num & (mask << 12)) << 12;
    result |= (num & (mask << 28)) >> 0;

    return result;
}

Output<Node> rearrange_constant(const Output<Node>& c, uint32_t groups) {
    auto constant = ov::as_type_ptr<v0::Constant>(c.get_node_shared_ptr());
    FRONT_END_OP_CONVERSION_CHECK(constant, "weight must be Constant.");
    FRONT_END_OP_CONVERSION_CHECK(constant->get_byte_size() == shape_size(constant->get_shape()) * sizeof(uint32_t),
                                  "AWQ constant storage size does not match expected int32 packing.");
    auto src = constant->get_data_ptr<uint32_t>();
    auto initial_shape = constant->get_shape();
    FRONT_END_OP_CONVERSION_CHECK(initial_shape.size() == 2, "Only 2D constants are supported.");
    FRONT_END_OP_CONVERSION_CHECK(groups > 0, "AWQ group size must be greater than 0.");
    FRONT_END_OP_CONVERSION_CHECK(initial_shape[0] % groups == 0,
                                  "AWQ qweight first dimension must be divisible by group size.");
    auto new_shape = Shape{initial_shape[0] / groups, groups, initial_shape[1] * 8};
    auto new_qweight = std::make_shared<v0::Constant>(element::u4, new_shape);
    auto dst = const_cast<uint32_t*>(reinterpret_cast<const uint32_t*>(new_qweight->get_data_ptr()));
    const size_t src_elements_count = shape_size(initial_shape);
    const size_t dst_elements_count = shape_size(new_shape) / 8;
    FRONT_END_OP_CONVERSION_CHECK(dst_elements_count == src_elements_count,
                                  "Unexpected AWQ constant size mismatch after rearrangement.");
    for (size_t i = 0; i < src_elements_count; i++) {
        dst[i] = rearrange_awq_bits(src[i]);
    }
    new_qweight->set_friendly_name(constant->get_friendly_name());
    return new_qweight;
}

// GPTQ packs 8 u4 values per int32 along the INPUT dimension.
// Each int32 at [i, j] holds u4 values for rows i*8..i*8+7 at column j.
// This is a transpose of the inner dims: [K, N, 8] u4 → [K, 8, N] u4.
Output<Node> unpack_gptq_qweight(const Output<Node>& c, int64_t group_size) {
    auto constant = ov::as_type_ptr<v0::Constant>(c.get_node_shared_ptr());
    FRONT_END_OP_CONVERSION_CHECK(constant, "qweight must be Constant.");
    FRONT_END_OP_CONVERSION_CHECK(constant->get_byte_size() == shape_size(constant->get_shape()) * sizeof(uint32_t),
                                  "GPTQ qweight storage size does not match expected int32 packing.");
    auto src = constant->get_data_ptr<uint32_t>();
    auto initial_shape = constant->get_shape();
    FRONT_END_OP_CONVERSION_CHECK(initial_shape.size() == 2, "Only 2D qweight constants are supported.");
    const size_t K = initial_shape[0];  // in_features / 8
    const size_t N = initial_shape[1];  // out_features
    const size_t in_features = K * 8;
    FRONT_END_OP_CONVERSION_CHECK(group_size > 0, "GPTQ group_size must be greater than 0.");
    FRONT_END_OP_CONVERSION_CHECK(static_cast<size_t>(group_size) <= in_features,
                                  "GPTQ group_size must not exceed in_features.");
    const size_t group_size_u = static_cast<size_t>(group_size);
    FRONT_END_OP_CONVERSION_CHECK(in_features % group_size_u == 0, "GPTQ in_features must be divisible by group_size.");
    const size_t n_groups = in_features / group_size_u;
    auto new_shape = Shape{n_groups, group_size_u, N};
    auto new_const = std::make_shared<v0::Constant>(element::u4, new_shape, 0);
    auto dst = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(new_const->get_data_ptr()));
    for (size_t i = 0; i < K; ++i) {
        for (size_t j = 0; j < N; ++j) {
            uint32_t val = src[i * N + j];
            for (size_t k = 0; k < 8; ++k) {
                set_u4(dst, (i * 8 + k) * N + j, (val >> (k * 4)) & 0xF);
            }
        }
    }
    new_const->set_friendly_name(constant->get_friendly_name());
    return new_const;
}

// GPTQ qzeros: packs 8 u4 values per int32 along the OUTPUT dimension.
// Byte layout matches u4 layout directly — single-pass copy with +1 offset.
Output<Node> unpack_gptq_qzeros(const Output<Node>& c) {
    auto constant = ov::as_type_ptr<v0::Constant>(c.get_node_shared_ptr());
    FRONT_END_OP_CONVERSION_CHECK(constant, "qzeros must be Constant.");
    FRONT_END_OP_CONVERSION_CHECK(constant->get_byte_size() == shape_size(constant->get_shape()) * sizeof(uint32_t),
                                  "GPTQ qzeros storage size does not match expected int32 packing.");
    auto src = reinterpret_cast<const uint8_t*>(constant->get_data_ptr());
    auto initial_shape = constant->get_shape();
    FRONT_END_OP_CONVERSION_CHECK(initial_shape.size() == 2, "Only 2D qzeros constants are supported.");
    auto new_shape = Shape{initial_shape[0], 1, initial_shape[1] * 8};
    auto new_const = std::make_shared<v0::Constant>(element::u4, new_shape);
    auto dst = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(new_const->get_data_ptr()));
    // Apply +1 offset per u4 value while copying (GPTQ stores zp-1)
    const size_t n_bytes = shape_size(initial_shape) * sizeof(uint32_t);
    for (size_t i = 0; i < n_bytes; ++i) {
        uint8_t lo = (src[i] & 0x0F) + 1;
        uint8_t hi = ((src[i] >> 4) & 0x0F) + 1;
        dst[i] = (lo & 0x0F) | static_cast<uint8_t>((hi & 0x0F) << 4);
    }
    new_const->set_friendly_name(constant->get_friendly_name());
    return new_const;
}

}  // namespace

OutputVector translate_linear_awq(const NodeContext& context) {
    num_inputs_check(context, 4, 7);
    auto x = context.get_input(0);
    auto qweight = context.get_input(1);
    auto qzeros = context.get_input(2);
    auto scales = context.get_input(3);
    auto groups = context.const_input<int64_t>(4);
    auto bits = context.const_input<int64_t>(5);

    FRONT_END_OP_CONVERSION_CHECK(bits == 4, "Only 4 bit AWQ is supported.");

    auto new_qweight = rearrange_constant(qweight, static_cast<uint32_t>(groups));
    auto new_qzeros = rearrange_constant(qzeros, 1);
    FRONT_END_OP_CONVERSION_CHECK(scales.get_partial_shape().is_static(), "Scales must be constant.");
    auto scales_shape = scales.get_shape();
    auto new_scales_shape =
        v0::Constant::create(element::i32, {3}, std::vector<uint64_t>{scales_shape[0], 1, scales_shape[1]});
    auto new_scales = context.mark_node(std::make_shared<v1::Reshape>(scales, new_scales_shape, false));
    auto out_shape =
        v0::Constant::create(element::i32, {2}, std::vector<int32_t>{static_cast<int32_t>(qweight.get_shape()[0]), -1});
    auto weight = low_precision_subgraph(context, x, new_qweight, new_qzeros, new_scales, out_shape);

    auto matmul = context.mark_node(std::make_shared<v0::MatMul>(x, weight, false, false));
    if (!context.input_is_none(6)) {
        auto bias = context.get_input(6);

        if (bias.get_element_type() == element::f16 || bias.get_element_type() == element::bf16) {
            bias = context.mark_node(std::make_shared<v1::ConvertLike>(bias, x));
        }
        matmul = context.mark_node(std::make_shared<v1::Add>(matmul, bias));
    }
    return {matmul};
};

OutputVector translate_linear_gptq(const NodeContext& context) {
    // ov_ext::gptq_gemm(input, qweight, qzeros, scales, group_size, bits, sym, bias?)
    num_inputs_check(context, 7, 8);
    auto x = context.get_input(0);
    auto qweight = context.get_input(1);
    auto qzeros = context.get_input(2);
    auto scales = context.get_input(3);
    auto group_size = context.const_input<int64_t>(4);
    auto bits = context.const_input<int64_t>(5);
    auto sym = context.const_input<bool>(6);

    FRONT_END_OP_CONVERSION_CHECK(bits == 4, "Only 4-bit GPTQ is supported.");

    // qweight: [in_features/8, out_features] int32 → [n_groups, group_size, out_features] u4
    auto new_qweight = unpack_gptq_qweight(qweight, group_size);

    Output<Node> new_qzeros;
    if (sym) {
        // Symmetric quantisation: zero point is always 8 (midpoint of u4 range)
        new_qzeros = context.mark_node(v0::Constant::create(element::u4, Shape{}, std::vector<uint8_t>{8}));
    } else {
        // qzeros: [n_groups, out_features/8] int32 → [n_groups, 1, out_features] u4 (with +1 offset)
        new_qzeros = unpack_gptq_qzeros(qzeros);
    }

    FRONT_END_OP_CONVERSION_CHECK(scales.get_partial_shape().is_static(), "Scales must be constant.");
    auto scales_shape = scales.get_shape();
    FRONT_END_OP_CONVERSION_CHECK(scales_shape.size() == 2,
                                  "GPTQ scales input is expected to be 2D, but got rank ",
                                  scales_shape.size(),
                                  ".");
    auto new_scales_shape =
        v0::Constant::create(element::i32, {3}, std::vector<uint64_t>{scales_shape[0], 1, scales_shape[1]});
    auto new_scales = context.mark_node(std::make_shared<v1::Reshape>(scales, new_scales_shape, false));
    // Reshape dequantised weight to [in_features, out_features] for matmul
    auto qweight_shape = qweight.get_shape();
    auto out_shape =
        v0::Constant::create(element::i32, {2}, std::vector<int32_t>{static_cast<int32_t>(qweight_shape[0] * 8), -1});
    auto weight = low_precision_subgraph(context, x, new_qweight, new_qzeros, new_scales, out_shape);

    auto matmul = context.mark_node(std::make_shared<v0::MatMul>(x, weight, false, false));
    if (!context.input_is_none(7)) {
        auto bias = context.get_input(7);

        if (bias.get_element_type() == element::f16 || bias.get_element_type() == element::bf16) {
            bias = context.mark_node(std::make_shared<v1::ConvertLike>(bias, x));
        }
        matmul = context.mark_node(std::make_shared<v1::Add>(matmul, bias));
    }
    return {matmul};
};

OutputVector translate_linear_bitnet(const NodeContext& context) {
    num_inputs_check(context, 3, 4);
    const auto x = context.get_input(0);
    const auto weight = context.get_input(1);
    const auto scales = context.get_input(2);

    const auto constant = ov::as_type_ptr<v0::Constant>(weight.get_node_shared_ptr());
    FRONT_END_OP_CONVERSION_CHECK(constant, "weight must be Constant.");
    const auto src = reinterpret_cast<const uint8_t*>(constant->get_data_ptr());
    const auto initial_shape = constant->get_shape();
    FRONT_END_OP_CONVERSION_CHECK(initial_shape.size() == 2, "Only 2D constants are supported.");
    const uint8_t values_per_pack = 4;  // Number of 2-bit values packed into a byte
    const size_t rows = initial_shape[0];
    const size_t cols = initial_shape[1];
    FRONT_END_OP_CONVERSION_CHECK(cols % values_per_pack == 0,
                                  "The second dimension of weight must be divisible by 4.");
    const auto new_shape = Shape{rows * values_per_pack, cols};
    auto new_weight = std::make_shared<v0::Constant>(element::u2, new_shape, 0);
    auto dst = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(new_weight->get_data_ptr()));
    const size_t row_len = cols / values_per_pack;
    // This lambda extracts 2 bits from each of 4 consecutive bytes at a given bit position,
    // then packs them into a single byte, placing each 2-bit value in its respective position (6, 4, 2, 0).
    const auto reorder_bitnet_2bit_values =
        [](const uint8_t* const src, const size_t src_idx, const size_t pos) -> uint8_t {
        const uint8_t values_per_byte = 4;
        const uint8_t value_mask = 0x3;
        const uint8_t value_size = 2;  // Size of each value is 2 bits
        uint8_t value{};               // Should be zeroed

        for (size_t value_idx = 0; value_idx != values_per_byte; ++value_idx) {
            value |= ((src[src_idx + value_idx] >> pos) & value_mask) << value_idx * value_size;
        }
        return value;
    };
    // In each 8bit value we have 4 2-bit values, first value contains first element, second value first element of a
    // next row. We need to repack them in contiguous way.
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < row_len; ++j) {
            const size_t src_idx = values_per_pack * j + i * cols;
            dst[j + (i + 0 * rows) * row_len] = reorder_bitnet_2bit_values(src, src_idx, 0);
            dst[j + (i + 1 * rows) * row_len] = reorder_bitnet_2bit_values(src, src_idx, 2);
            dst[j + (i + 2 * rows) * row_len] = reorder_bitnet_2bit_values(src, src_idx, 4);
            dst[j + (i + 3 * rows) * row_len] = reorder_bitnet_2bit_values(src, src_idx, 6);
        }
    }
    new_weight->set_friendly_name(constant->get_friendly_name());
    const auto zero_point = context.mark_node(std::make_shared<v0::Constant>(element::u2, Shape{}, 1));
    auto mm_weight = low_precision_subgraph(context, x, new_weight, zero_point, scales, {});

    auto matmul = context.mark_node(std::make_shared<v0::MatMul>(x, mm_weight, false, true));
    if (!context.input_is_none(3)) {
        auto bias = context.get_input(3);

        if (bias.get_element_type() == element::f16 || bias.get_element_type() == element::bf16) {
            bias = context.mark_node(std::make_shared<v1::ConvertLike>(bias, x));
        }
        matmul = context.mark_node(std::make_shared<v1::Add>(matmul, bias));
    }
    return {matmul};
};

OutputVector translate_bmm_ext(const NodeContext& context) {
    // ov_ext::bmm - batch matrix multiplication for 16-bit models
    // schema: ov_ext::bmm(Tensor batch1, Tensor batch2) -> Tensor
    num_inputs_check(context, 2, 2);
    auto batch1 = context.get_input(0);
    auto batch2 = context.get_input(1);
    const auto initial_batch1 = batch1;

    // Handle mixed precision - convert to f32 if inputs are fp16/bf16
    const bool convert_back = batch1.get_element_type() != element::f32;
    if (batch2.get_element_type() != element::f32) {
        batch2 = context.mark_node(std::make_shared<v0::Convert>(batch2, element::f32));
    }
    if (convert_back) {
        batch1 = context.mark_node(std::make_shared<v0::Convert>(batch1, element::f32));
    }

    // bmm: (b, n, m) @ (b, m, p) -> (b, n, p)
    auto matmul = context.mark_node(std::make_shared<v0::MatMul>(std::move(batch1), std::move(batch2), false, false));

    if (convert_back) {
        matmul = context.mark_node(std::make_shared<v1::ConvertLike>(std::move(matmul), initial_batch1));
    }
    return {matmul};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
