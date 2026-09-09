// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cmath>
#include <numeric>

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/decompositions/low_precision_dequantize.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {
namespace {
// MatMulNBits repacks B/scales/zero_point initializers into new low-bit Constants. Preserve the
// original ONNX initializer's friendly name and tensor names on the repacked Constant so weight
// sharing / weights-as-input can still identify it as the same external weight.
void preserve_initializer_name(const std::shared_ptr<v0::Constant>& repacked,
                               const std::shared_ptr<v0::Constant>& original) {
    repacked->set_friendly_name(original->get_friendly_name());
    repacked->get_output_tensor(0).set_names(original->get_output_tensor(0).get_names());
}
}  // namespace

ov::OutputVector matmulnbits(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 3);
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulNBits
    const auto inputs = node.get_ov_inputs();
    const auto& a = inputs[0];  // required
    ov::Output<ov::Node> b;
    const auto& b_quantized = inputs[1];                                                 // required
    const auto& scales = inputs[2];                                                      // required
    ov::Output<ov::Node> zero_points;                                                    // optional, input[3]
    ov::Output<ov::Node> group_idx;                                                      // optional, input[4]
    ov::Output<ov::Node> bias;                                                           // optional, input[5]
    const auto K = node.get_attribute_value<int64_t>("K");                               // required
    const auto N = node.get_attribute_value<int64_t>("N");                               // required
    const auto accuracy_level = node.get_attribute_value<int64_t>("accuracy_level", 0);  // optional, default unset(0)
    const auto block_size = node.get_attribute_value<int64_t>("block_size");             // required
    const auto bits = node.get_attribute_value<int64_t>(
        "bits",
        4);  // required, in docs: number of bits used for weight quantization (default 4)
    // optional, default 0 (not prepacked). 1/2 mean B is prepacked into a CUDA-specific
    // (CUTLASS SM80/SM90 fpA_intB) byte layout that this frontend does not decode.
    const auto weight_prepacked = node.get_attribute_value<int64_t>("weight_prepacked", 0);

    // Validate attributes before any arithmetic to prevent division-by-zero and signed overflow
    CHECK_VALID_NODE(node, K > 0, "Wrong K attribute value: ", K);
    CHECK_VALID_NODE(node, N > 0, "Wrong N attribute value: ", N);
    CHECK_VALID_NODE(node,
                     block_size >= 16 && (block_size & (block_size - 1)) == 0,
                     "Wrong block size, should be >=16 and be a power of 2, got: ",
                     block_size);
    CHECK_VALID_NODE(node, bits == 2 || bits == 4 || bits == 8, "Unsupported bits value: ", bits);
    CHECK_VALID_NODE(node, accuracy_level >= 0 && accuracy_level <= 4, "Unsupported accuracy level: ", accuracy_level);
    CHECK_VALID_NODE(node,
                     weight_prepacked == 0,
                     "MatMulNBits limitation: weight_prepacked != 0 requires decoding an EP-specific "
                     "(CUDA CUTLASS) prepacked weight layout, which is not supported, got: ",
                     weight_prepacked);

    const auto u_K = static_cast<uint64_t>(K);
    const auto u_block_size = static_cast<uint64_t>(block_size);
    const uint64_t n_blocks_per_col = (u_K + u_block_size - 1) / u_block_size;
    const auto blob_size = (block_size * bits + 7) / 8;

    const uint64_t expected_b_size = N * n_blocks_per_col * blob_size;
    const auto& b_shape = b_quantized.get_partial_shape();

    CHECK_VALID_NODE(node, n_blocks_per_col > 0, "Wrong blocks count: ", n_blocks_per_col);
    CHECK_VALID_NODE(node, blob_size > 0, "Wrong blob size: ", blob_size);
    // in documentation: ...Input B is a 2D constant Matrix.
    CHECK_VALID_NODE(node,
                     ov::as_type<v0::Constant>(b_quantized.get_node()) != nullptr,
                     "MatMulNBits limitation: accepting only a constant as a B input");
    CHECK_VALID_NODE(
        node,
        b_shape.is_static(),
        "Expected input B shape is static and compatible with shape [N][n_blocks_per_col][blob_size], got: ",
        b_shape);
    const auto& b_shape_static = b_shape.get_shape();
    const uint64_t actual_b_size =
        std::accumulate(b_shape_static.begin(), b_shape_static.end(), uint64_t{1}, std::multiplies<uint64_t>{});
    CHECK_VALID_NODE(
        node,
        actual_b_size == expected_b_size,
        "Expected input B shape is static and compatible with shape [N][n_blocks_per_col][blob_size], got: ",
        b_shape);
    CHECK_VALID_NODE(node,
                     a.get_element_type() == ov::element::f16 || a.get_element_type() == ov::element::f32 ||
                         a.get_element_type() == ov::element::bf16 || a.get_element_type() == ov::element::dynamic,
                     "Unsupported input A type, accepted dynamic, FP16, FP32, BF16, got: ",
                     a.get_element_type());
    CHECK_VALID_NODE(
        node,
        b_quantized.get_element_type() == ov::element::u8 || b_quantized.get_element_type() == ov::element::i32,
        "Unsupported input B type, accepted U8 or I32, got: ",
        b_quantized.get_element_type());

    if (common::is_input_valid(node, 3)) {
        zero_points = inputs[3];
        CHECK_VALID_NODE(node,
                         zero_points.get_element_type() == ov::element::u8 ||
                             zero_points.get_element_type() == ov::element::i32 ||
                             zero_points.get_element_type() == ov::element::f32 ||
                             zero_points.get_element_type() == ov::element::f16 ||
                             zero_points.get_element_type() == ov::element::bf16,
                         "Unsupported input zero_points type, accepted U8, I32, FP16, FP32, BF16, got: ",
                         zero_points.get_element_type());
    }

    if (common::is_input_valid(node, 4)) {
        group_idx = inputs[4];
        CHECK_VALID_NODE(node,
                         group_idx.get_element_type() == ov::element::i32,
                         "Unsupported input group_idx type, accepted I32, got: ",
                         group_idx.get_element_type());
        // group_idx[k] gives the block/group index dequantizing element k should use, replacing the
        // usual floor(k / block_size) assignment (act-order/GPTQ-style non-sequential grouping).
        // Per onnxruntime's own validator (matmul_nbits_helper.h): 1D, sized K or block-padded K.
        const auto& g_idx_shape = group_idx.get_partial_shape();
        CHECK_VALID_NODE(node,
                         g_idx_shape.rank().is_static() && g_idx_shape.size() == 1 && g_idx_shape[0].is_static(),
                         "Expected input group_idx to be a 1D tensor with a static size, got: ",
                         g_idx_shape);
        const auto g_idx_size = static_cast<uint64_t>(g_idx_shape[0].get_length());
        const auto k_padded = n_blocks_per_col * u_block_size;
        CHECK_VALID_NODE(node,
                         g_idx_size == u_K || g_idx_size == k_padded,
                         "Wrong group_idx size, expected K (",
                         K,
                         ") or block-padded K (",
                         k_padded,
                         "), got: ",
                         g_idx_size);
        // group_idx is consumed as a Gather index below; OV's Gather silently normalizes negative
        // indices and zero-fills out-of-range ones instead of throwing, so an invalid model would
        // otherwise import and dequantize with the wrong scale/zero-point. Require it constant (same
        // convention as B/zero_points above) and validate every value is in [0, n_blocks_per_col).
        CHECK_VALID_NODE(node,
                         ov::as_type<v0::Constant>(group_idx.get_node()) != nullptr,
                         "MatMulNBits limitation: accepting only a constant as a group_idx");
        const auto group_idx_const = ov::as_type_ptr<v0::Constant>(group_idx.get_node_shared_ptr());
        const auto group_idx_values = group_idx_const->cast_vector<int32_t>();
        const auto minmax = std::minmax_element(group_idx_values.begin(), group_idx_values.end());
        CHECK_VALID_NODE(node,
                         *minmax.first >= 0 && *minmax.second < static_cast<int32_t>(n_blocks_per_col),
                         "group_idx values must be within [0, n_blocks_per_col=",
                         n_blocks_per_col,
                         "), got range [",
                         *minmax.first,
                         ", ",
                         *minmax.second,
                         "]");
    }

    if (common::is_input_valid(node, 5)) {
        bias = inputs[5];
        CHECK_VALID_NODE(node,
                         bias.get_element_type() == a.get_element_type() ||
                             a.get_element_type() == ov::element::dynamic ||
                             bias.get_element_type() == ov::element::dynamic,
                         "Unsupported input bias type, must be equal to input A type, got: ",
                         bias.get_element_type());
        CHECK_VALID_NODE(node,
                         bias.get_partial_shape() == PartialShape{N},
                         "Wrong bias shape, expected [",
                         N,
                         "], got: ",
                         bias.get_partial_shape());
    }
    const auto zero = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, 0);
    const auto one = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, 1);
    const auto elements = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, static_cast<int32_t>(K));
    const auto axis = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, 1);

    ov::Output<ov::Node> mm_output;
    {
        const auto b_const = ov::as_type_ptr<v0::Constant>(b_quantized.get_node_shared_ptr());

        ov::Output<ov::Node> casted_b;
        ov::Shape casted_b_shape;
        ov::Output<ov::Node> default_zp;
        ov::element::Type zp_element_type;
        // Casting/converting data of source constant.
        // For further calculations (sub and/or multiply) we need to reshape
        // b -> [N][n_blocks_per_col][block_size]
        switch (bits) {
        case 2:
            casted_b_shape = ov::Shape{static_cast<size_t>(N),
                                       static_cast<size_t>(n_blocks_per_col),
                                       static_cast<size_t>(blob_size * 4)};
            casted_b = std::make_shared<v0::Constant>(ov::element::u2, casted_b_shape, b_const->get_data_ptr());
            default_zp = std::make_shared<v0::Constant>(ov::element::u2, Shape{1}, 2);
            zp_element_type = ov::element::u2;
            break;
        case 4:
            casted_b_shape = ov::Shape{static_cast<size_t>(N),
                                       static_cast<size_t>(n_blocks_per_col),
                                       static_cast<size_t>(blob_size * 2)};
            casted_b = std::make_shared<v0::Constant>(ov::element::u4, casted_b_shape, b_const->get_data_ptr());
            default_zp = std::make_shared<v0::Constant>(ov::element::u4, Shape{1}, 8);
            zp_element_type = ov::element::u4;
            break;
        case 8:
            casted_b_shape = ov::Shape{static_cast<size_t>(N),
                                       static_cast<size_t>(n_blocks_per_col),
                                       static_cast<size_t>(blob_size)};
            casted_b = std::make_shared<v0::Constant>(ov::element::u8, casted_b_shape, b_const->get_data_ptr());
            default_zp = std::make_shared<v0::Constant>(ov::element::u8, Shape{1}, 128);
            zp_element_type = ov::element::u8;
            break;
        default:
            FRONT_END_THROW("Unsupported bits count");
            break;
        }

        // Preserve the original B initializer name on the repacked weight constant so it can still be
        // identified by name downstream (e.g. for weight sharing).
        if (const auto casted_b_const = ov::as_type_ptr<v0::Constant>(casted_b.get_node_shared_ptr())) {
            preserve_initializer_name(casted_b_const, b_const);
        }

        ov::Output<ov::Node> converted_zero_points;
        if (!zero_points.get_node_shared_ptr()) {
            converted_zero_points = std::make_shared<v0::Convert>(default_zp, a.get_element_type());
        } else {
            // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulNBits
            // according to the link, zero point are:
            // Constrain quantized zero point types to uint8/int32/float16/float.
            // If zero_points has same type as A
            //   it's not packed and has the same shape as Scales. [N * n_blocks_per_col]
            // If zero_points is stored as uint8_t.
            //   It has the same packing method as input B. [N * CeilDiv(n_blocks_per_col * bits, 8)]
            //
            // no matter which package method, the outputs of this section will be
            //   {A type, [N, n_blocks_per_col, 1]}
            CHECK_VALID_NODE(node,
                             ov::as_type<v0::Constant>(zero_points.get_node()) != nullptr,
                             "MatMulNBits limitation: accepting only a constant as a zero_points");

            const auto zero_points_const = ov::as_type_ptr<v0::Constant>(zero_points.get_node_shared_ptr());
            if (zero_points.get_element_type() == a.get_element_type()) {
                const uint64_t expected_zp_size = N * n_blocks_per_col;
                const auto& zp_shape = zero_points.get_partial_shape();
                CHECK_VALID_NODE(node,
                                 zp_shape.is_static(),
                                 "Expected input Zero Point shape is static and compatible with shape "
                                 "[N][n_blocks_per_col][1], got: ",
                                 zp_shape);

                const auto zp_shape_static = zp_shape.get_shape();
                uint64_t actual_zp_size = 1;
                for (const auto dim : zp_shape_static) {
                    actual_zp_size *= dim;
                }
                CHECK_VALID_NODE(node,
                                 actual_zp_size == expected_zp_size,
                                 "Expected input Zero Point shape is compatible with shape [N][n_blocks_per_col][1], "
                                 "got: ",
                                 zp_shape);

                ov::Shape casted_zp_shape = ov::Shape{static_cast<size_t>(N), static_cast<size_t>(n_blocks_per_col), 1};
                converted_zero_points = std::make_shared<v0::Constant>(a.get_element_type(),
                                                                       casted_zp_shape,
                                                                       zero_points_const->get_data_ptr());
                // Preserve the original zero_point name on the repacked Constant (as done for B) so weight
                // sharing / weights-as-input can still identify it by name.
                if (const auto casted_zp_const =
                        ov::as_type_ptr<v0::Constant>(converted_zero_points.get_node_shared_ptr())) {
                    preserve_initializer_name(casted_zp_const, zero_points_const);
                }
            } else if (zero_points.get_element_type() == ov::element::u8) {
                // for alignment, n_blocks_per_col might not aligned to num_per_byte
                uint64_t num_per_byte = 8 / bits;
                uint64_t num_byte = (n_blocks_per_col + (num_per_byte - 1)) / num_per_byte;
                uint64_t num_elements_aligned = num_byte * num_per_byte;
                ov::Shape casted_zp_shape =
                    ov::Shape{static_cast<size_t>(N), static_cast<size_t>(num_elements_aligned), 1};
                auto casted_zp_org =
                    std::make_shared<v0::Constant>(zp_element_type, casted_zp_shape, zero_points_const->get_data_ptr());
                // Preserve the original zero_point name on the repacked Constant (as done for B) so weight
                // sharing can promote it. The packed Constant keeps the source uint8 byte count, so the
                // promoted input's tensor matches the external weight at runtime - true whether or not the
                // Slice below is inserted, so name it unconditionally.
                preserve_initializer_name(casted_zp_org, zero_points_const);
                converted_zero_points = std::make_shared<v0::Convert>(casted_zp_org, a.get_element_type());
                if (n_blocks_per_col != num_elements_aligned) {
                    // if not align
                    // for example, n_blocks_per_col is 13, bits is 2, num_per_byte is 4, it will packed into 4 bytes
                    // need to make a constant: uint2, {N, 16}
                    // then slice to: uint2, {N, 13}
                    const auto num_elements = std::make_shared<v0::Constant>(ov::element::i32,
                                                                             Shape{1},
                                                                             static_cast<int32_t>(n_blocks_per_col));
                    converted_zero_points =
                        std::make_shared<v8::Slice>(converted_zero_points, zero, num_elements, one, axis);
                }
            } else {
                FRONT_END_THROW("Unexpected zero point type");
            }
        }

        // OV core has no Slice evaluate()/constant-fold path for packed sub-byte types (u4/u2/i4/nf4/
        // f4e2m1) by design (CVS-173497, PR #32388 "[Core/Op] Disable Slice constant folding and
        // evaluate for LP"), so casted_b must be dequantized to a.get_element_type() first; any Slice
        // (padding trim) has to run after that conversion, never directly on the raw packed data.

        // compute in input A's precision (FP32/FP16/BF16)
        const auto scales_converted = std::make_shared<v0::Convert>(scales, a.get_element_type());

        if (group_idx.get_node_shared_ptr()) {
            // group_idx present: element k's scale/zero-point comes from block group_idx[k], not
            // floor(k / block_size). There is no shared contiguous block left to broadcast over, so
            // flatten B/scales/zero_points along the block axis and Gather each K-column's group
            // directly (this mirrors onnxruntime's own reorder_idx dequant path, which is likewise a
            // per-element lookup rather than a block-broadcast fast path).
            auto flat_shape = v0::Constant::create(ov::element::i32, ov::Shape{2}, {0, -1});
            ov::Output<ov::Node> b_flat = std::make_shared<v1::Reshape>(casted_b, flat_shape, true);  // [N, k_padded]

            const auto k_padded = n_blocks_per_col * u_block_size;
            const auto g_idx_size = static_cast<uint64_t>(group_idx.get_partial_shape()[0].get_length());
            const bool g_idx_is_full_k = (g_idx_size == u_K);
            // group_idx sized K: trim B to K first so it lines up with group_idx for the gather below.
            // group_idx sized block-padded K: gather stays block-padded; trim to K (if any) happens
            // after dequantization instead (see below).
            if (g_idx_is_full_k && k_padded != u_K) {
                b_flat = std::make_shared<v8::Slice>(b_flat, zero, elements, one, axis);
            }

            const auto gather_axis = v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            auto scales_flat =
                op::util::reshape(scales_converted,
                                  ov::Shape{static_cast<size_t>(N), static_cast<size_t>(n_blocks_per_col)});
            ov::Output<ov::Node> scale_per_k = std::make_shared<v8::Gather>(scales_flat, group_idx, gather_axis);

            ov::Output<ov::Node> zp_per_k = converted_zero_points;
            if (zero_points.get_node_shared_ptr()) {
                // converted_zero_points is [N, n_blocks_per_col, 1] here; drop the trailing 1 to gather.
                auto zp_flat =
                    op::util::reshape(converted_zero_points,
                                      ov::Shape{static_cast<size_t>(N), static_cast<size_t>(n_blocks_per_col)});
                zp_per_k = std::make_shared<v8::Gather>(zp_flat, group_idx, gather_axis);
            }
            // else: converted_zero_points already holds the default zp as a [1]-shaped scalar, which
            // broadcasts naturally against the [N, len(group_idx)] tensors above.

            auto dequant = ov::decomposition::low_precision_dequantize(b_flat, scale_per_k, zp_per_k);
            if (!g_idx_is_full_k && k_padded != u_K) {
                b = std::make_shared<v8::Slice>(dequant, zero, elements, one, axis);
            } else {
                b = dequant;
            }
        } else {
            // sub and scale via the shared low-precision dequantization helper
            const auto scales_reshaped =
                op::util::reshape(scales_converted,
                                  ov::Shape{static_cast<size_t>(N), static_cast<size_t>(n_blocks_per_col), 1});

            auto scaled_b =
                ov::decomposition::low_precision_dequantize(casted_b, scales_reshaped, converted_zero_points);

            // reshape b to [N, K]
            auto shape_b = v0::Constant::create(ov::element::i32, ov::Shape{2}, {0, -1});
            auto reshaped_b = std::make_shared<v1::Reshape>(scaled_b, shape_b, true);

            // if n_blocks_per_col*blob_size*X != K
            // need slice it to K
            // to produce b = [N, K]
            const bool slice_needed = (K % block_size != 0);
            if (slice_needed) {
                b = std::make_shared<v8::Slice>(reshaped_b, zero, elements, one, axis);
            } else {
                b = reshaped_b;
            }
        }

        // mm = matmul(a,b)
        mm_output = std::make_shared<v0::MatMul>(a, b, false, true);
    }

    if (bias.get_node_shared_ptr()) {
        return {std::make_shared<v1::Add>(mm_output, bias)};
    } else {
        return {mm_output};
    }
}

ONNX_OP("MatMulNBits", OPSET_SINCE(1), com_microsoft::opset_1::matmulnbits, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
