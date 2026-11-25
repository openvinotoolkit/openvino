#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "openvino/core/parallel.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/decompositions/low_precision_dequantize.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/runtime/tensor.hpp"
#include "ov_ops/fully_connected_compressed.hpp"

namespace ov::util {

inline void unpack_gguf_q8_0(const uint8_t* payload, size_t byte_size, Tensor& weights, Tensor& scales) {
    constexpr size_t block_elements = 32;
    constexpr size_t block_bytes = 34;
    const auto blocks = scales.get_size();
    OPENVINO_ASSERT(byte_size % block_bytes == 0 && byte_size / block_bytes == blocks &&
                        weights.get_size() % block_elements == 0 && weights.get_size() / block_elements == blocks,
                    "Invalid GGUF Q8_0 payload or output size");
    auto* codes = weights.data<int8_t>();
    auto* factors = scales.data<float16>();
    parallel_for(blocks, [&](size_t index) {
        const auto* block = payload + index * block_bytes;
        factors[index] = float16::from_bits(static_cast<uint16_t>(block[0] | (block[1] << 8)));
        std::memcpy(codes + index * block_elements, block + 2, block_elements);
    });
}

inline std::shared_ptr<Node> make_gguf_q8_0_fc_compatible(const op::internal::FullyConnectedCompressed& fc) {
    const auto weight = as_type_ptr<op::v0::Constant>(fc.get_input_node_shared_ptr(1));
    if (!weight || weight->get_element_type() != element::gguf_q8_0 ||
        fc.get_input_element_type(0) != element::f32 || fc.get_output_element_type(0) != element::f32) {
        return {};
    }
    for (size_t port = 2; port < fc.get_input_size(); ++port) {
        const auto absent = as_type_ptr<op::v0::Constant>(fc.get_input_node_shared_ptr(port));
        if (!absent || shape_size(absent->get_shape()) != 0) {
            return {};
        }
    }
    const auto& shape = weight->get_shape();
    constexpr size_t group_size = 32;
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] && shape[1] && shape[1] % group_size == 0,
                    "Invalid GGUF Q8_0 FC weight shape: ", shape);
    const auto groups = shape[1] / group_size;
    Tensor codes(element::i8, {shape[0], groups, group_size});
    Tensor scales(element::f16, {shape[0], groups, 1});
    unpack_gguf_q8_0(static_cast<const uint8_t*>(weight->get_data_ptr()), weight->get_byte_size(), codes, scales);
    const auto code_node = std::make_shared<op::v0::Constant>(codes);
    const auto scale_node = std::make_shared<op::v0::Constant>(scales);
    const auto target = std::make_shared<op::v0::Constant>(element::i64, Shape{shape.size()}, shape);
    const auto dequantized = decomposition::low_precision_dequantize(code_node, scale_node, {}, target);
    const auto converted = std::make_shared<op::v0::Convert>(dequantized, element::f32);
    return std::make_shared<op::v0::MatMul>(fc.input_value(0), converted, false, true);
}

}  // namespace ov::util