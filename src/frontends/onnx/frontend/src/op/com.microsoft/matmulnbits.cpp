// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cmath"
#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
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

    const uint64_t n_blocks_per_col = (K + block_size - 1) / block_size;
    const auto blob_size = (block_size * bits + 7) / 8;

    const uint64_t expected_b_size = N * n_blocks_per_col * blob_size;
    const auto& b_shape = b_quantized.get_partial_shape();
    uint64_t actual_b_size = 1;
    for (const auto& d : b_shape) {
        actual_b_size *= d.get_length();
    }

    CHECK_VALID_NODE(node, n_blocks_per_col > 0, "Wrong blocks count: ", n_blocks_per_col);
    CHECK_VALID_NODE(node, blob_size > 0, "Wrong blob size: ", blob_size);
    // in documentation: ...Input B is a 2D constant Matrix.
    CHECK_VALID_NODE(node,
                     ov::as_type<v0::Constant>(b_quantized.get_node()) != nullptr,
                     "MatMulNBits limitation: accepting only a constant as a B input");
    CHECK_VALID_NODE(
        node,
        b_shape.is_static() && actual_b_size == expected_b_size,
        "Expected input B shape is static and compatible with shape [N][n_blocks_per_col][blob_size], got: ",
        b_shape);
    CHECK_VALID_NODE(node,
                     a.get_element_type() == ov::element::f16 || a.get_element_type() == ov::element::f32 ||
                         a.get_element_type() == ov::element::dynamic,
                     "Unsupported input A type, accepted dynamic, FP16, FP32, got: ",
                     a.get_element_type());
    CHECK_VALID_NODE(
        node,
        b_quantized.get_element_type() == ov::element::u8 || b_quantized.get_element_type() == ov::element::i32,
        "Unsupported input B type, accepted FP16, FP32, got: ",
        b_quantized.get_element_type());

    CHECK_VALID_NODE(node,
                     block_size >= 16 && (block_size % 2 == 0),
                     "Wrong block size, should be >=16 and be a power of 2, got: ",
                     block_size);
    CHECK_VALID_NODE(node, accuracy_level >= 0 && accuracy_level <= 4, "Unsupported accuracy level: ", accuracy_level);

    if (inputs.size() > 3 && !ov::as_type_ptr<NullNode>(inputs[3].get_node_shared_ptr())) {
        zero_points = inputs[3];
        std::cout << "zero_points: " << zero_points << std::endl;
        CHECK_VALID_NODE(node,
                         zero_points.get_element_type() == ov::element::u8 ||
                             zero_points.get_element_type() == ov::element::i32 ||
                             zero_points.get_element_type() == ov::element::f32 ||
                             zero_points.get_element_type() == ov::element::f16,
                         "Unsupported input zero_points type, accepted U8, I32, FP16, FP32, got: ",
                         zero_points.get_element_type());
    }

    if (inputs.size() > 4 && !ov::as_type_ptr<NullNode>(inputs[4].get_node_shared_ptr())) {
        group_idx = inputs[4];
        CHECK_VALID_NODE(node,
                         group_idx.get_element_type() == ov::element::i32,
                         "Unsupported input group_idx type, accepted I32, got: ",
                         group_idx.get_element_type());
    }

    if (inputs.size() > 5 && !ov::as_type_ptr<NullNode>(inputs[5].get_node_shared_ptr())) {
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
                const uint64_t expected_zp_size = N * n_blocks_per_col * 1;
                const auto& zp_shape = zero_points.get_partial_shape();
                uint64_t actual_zp_size = 1;
                for (const auto& d : zp_shape) {
                    actual_zp_size *= d.get_length();
                }
                CHECK_VALID_NODE(node,
                                 zp_shape.is_static() && actual_zp_size == expected_zp_size,
                                 "Expected input Zero Point shape is static and compatible with shape "
                                 "[N][n_blocks_per_col][1], got: ",
                                 zp_shape);

                ov::Shape casted_zp_shape = ov::Shape{static_cast<size_t>(N), static_cast<size_t>(n_blocks_per_col), 1};
                converted_zero_points = std::make_shared<v0::Constant>(a.get_element_type(),
                                                                       casted_zp_shape,
                                                                       zero_points_const->get_data_ptr());
            } else if (zero_points.get_element_type() == ov::element::u8) {
                // for alignment, n_blocks_per_col might not aligned to num_per_byte
                uint64_t num_per_byte = 8 / bits;
                uint64_t num_byte = (n_blocks_per_col + (num_per_byte - 1)) / num_per_byte;
                uint64_t num_elements_aligned = num_byte * num_per_byte;
                ov::Shape casted_zp_shape =
                    ov::Shape{static_cast<size_t>(N), static_cast<size_t>(num_elements_aligned), 1};
                auto casted_zp_org =
                    std::make_shared<v0::Constant>(zp_element_type, casted_zp_shape, zero_points_const->get_data_ptr());
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

        // Possible issue with slice implementation, had to move convertion before slice, instead of slicing uint4
        // TODO: Ticket
        // Comments: it is still there, so need to convert b to fp16 first.

        // TODO: Need to collect performance data in case constant folding is applied. Possible some perf/mem-gap
        // Comments: in this latest code, the const folding is gone, it trigle the oneDNN kernel
        //           and use u2/u4/u8 weights as the kernel's input, won't do const folding anymore.

        // use fp16 for compute

        // convert b to fp16
        auto converted_b = std::make_shared<v0::Convert>(casted_b, a.get_element_type());

        // sub and scale
        const auto sub_b = std::make_shared<v1::Subtract>(converted_b, converted_zero_points);
        const auto scales_fp16 = std::make_shared<v0::Convert>(scales, a.get_element_type());
        const auto scales_reshaped =
            op::util::reshape(scales_fp16, ov::Shape{static_cast<size_t>(N), static_cast<size_t>(n_blocks_per_col), 1});
        const auto scaled_b = std::make_shared<v1::Multiply>(sub_b, scales_reshaped);

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
