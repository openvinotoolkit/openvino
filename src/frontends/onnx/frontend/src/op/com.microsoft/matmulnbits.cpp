// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cmath"
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
    const auto blob_size = static_cast<int64_t>(ceil(block_size * bits / 8));

    CHECK_VALID_NODE(node, n_blocks_per_col > 0, "Wrong blocks count: ", n_blocks_per_col);
    CHECK_VALID_NODE(node, blob_size > 0, "Wrong blob size: ", blob_size);
    // in documentation: ...Input B is a 2D constant Matrix.
    CHECK_VALID_NODE(node,
                     ov::as_type<v0::Constant>(b_quantized.get_node()) != nullptr,
                     "MatMulNBits limitation: accepting only a constant as a B input");
    CHECK_VALID_NODE(node,
                     b_quantized.get_partial_shape().rank() == 3,
                     "Expected rank of quantized weights is 3 [N][n_blocks_per_col][blob_size], got: ",
                     b_quantized.get_partial_shape().rank());
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

    if (inputs.size() > 3) {
        zero_points = inputs[3];
        CHECK_VALID_NODE(node,
                         zero_points.get_element_type() == ov::element::u8 ||
                             zero_points.get_element_type() == ov::element::i32 ||
                             zero_points.get_element_type() == ov::element::f32 ||
                             zero_points.get_element_type() == ov::element::f16,
                         "Unsupported input zero_points type, accepted U8, I32, FP16, FP32, got: ",
                         zero_points.get_element_type());
    }

    if (inputs.size() > 4) {
        group_idx = inputs[4];
        CHECK_VALID_NODE(node,
                         group_idx.get_element_type() == ov::element::i32,
                         "Unsupported input group_idx type, accepted I32, got: ",
                         group_idx.get_element_type());
    }

    if (inputs.size() > 5) {
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

    ov::Output<ov::Node> mm_output;
    {
        const auto b_const = ov::as_type_ptr<v0::Constant>(b_quantized.get_node_shared_ptr());

        ov::Output<ov::Node> casted_b;
        ov::Shape casted_b_shape;
        ov::Output<ov::Node> default_zp;
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
            break;
        case 4:
            casted_b_shape = ov::Shape{static_cast<size_t>(N),
                                       static_cast<size_t>(n_blocks_per_col),
                                       static_cast<size_t>(blob_size * 2)};
            casted_b = std::make_shared<v0::Constant>(ov::element::u4, casted_b_shape, b_const->get_data_ptr());
            default_zp = std::make_shared<v0::Constant>(ov::element::u4, Shape{1}, 8);
            break;
        case 8:
            casted_b_shape = ov::Shape{static_cast<size_t>(N),
                                       static_cast<size_t>(n_blocks_per_col),
                                       static_cast<size_t>(blob_size)};
            casted_b = std::make_shared<v0::Constant>(ov::element::u8, casted_b_shape, b_const->get_data_ptr());
            default_zp = std::make_shared<v0::Constant>(ov::element::u8, Shape{1}, 128);
            break;
        default:
            FRONT_END_THROW("Unsupported bits count");
            break;
        }

        if (!zero_points.get_node_shared_ptr()) {
            zero_points = default_zp;
        } else {
            // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.MatMulNBits
            // according to the link, zero point are:
            // Constrain quantized zero point types to uint8/int32/float16/float.
            // Input zero_points is stored as uint8_t or same as type(A). It has the same packing method as input B
            zero_points =
                op::util::reshape(zero_points,
                                  ov::Shape{static_cast<size_t>(N), static_cast<size_t>(n_blocks_per_col), 1});
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
        auto converted_zero_points = std::make_shared<v0::Convert>(zero_points, a.get_element_type());

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
            const auto zero = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, 0);
            const auto one = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, 1);
            const auto elements = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, static_cast<int32_t>(K));
            const auto axis = std::make_shared<v0::Constant>(ov::element::i32, Shape{1}, 1);
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