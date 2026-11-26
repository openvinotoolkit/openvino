#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/decompositions/low_precision_dequantize.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/runtime/tensor.hpp"
#include "ov_ops/fully_connected_compressed.hpp"

namespace ov::util {

inline uint8_t quantize_gguf_zp_u8(float zpval) {
    OPENVINO_ASSERT(std::isfinite(zpval), "[GGUF] cannot quantize a non-finite zero-point");
    return static_cast<uint8_t>(std::clamp(std::lround(zpval), 0L, 255L));
}

inline void requantize_gguf_group_u4(const uint8_t* source_q,
                                     float sc,
                                     float offset,
                                     ov::float16& out_scale,
                                     uint8_t& out_zp,
                                     uint8_t* out_packed) {
    float values[32];
    float min_value = 0.0f;
    float max_value = 0.0f;
    for (int k = 0; k < 32; ++k) {
        values[k] = sc * static_cast<float>(source_q[k]) - offset;
        min_value = std::min(min_value, values[k]);
        max_value = std::max(max_value, values[k]);
    }

    struct Error {
        double squared;
        float max_abs;
    };
    const auto measure = [&](float runtime_scale, uint8_t zp, const uint8_t* quantized) {
        Error error{0.0, 0.0f};
        for (int k = 0; k < 32; ++k) {
            const float diff = values[k] - runtime_scale * (static_cast<int>(quantized[k]) - zp);
            error.squared += static_cast<double>(diff) * diff;
            error.max_abs = std::max(error.max_abs, std::fabs(diff));
        }
        return error;
    };

    uint8_t best_q[32];
    std::copy_n(source_q, 32, best_q);
    ov::float16 best_scale(sc);
    uint8_t best_zp = quantize_gguf_zp_u8(sc != 0.f ? offset / sc : 0.f);
    Error best_error = measure(static_cast<float>(best_scale), best_zp, best_q);
    const float fallback_max_error = best_error.max_abs;

    float initial_scale = (max_value - min_value) / 15.0f;
    if (std::fabs(initial_scale) < std::numeric_limits<float>::epsilon()) {
        initial_scale = std::numeric_limits<float>::epsilon();
    }
    const auto rounded_zp = static_cast<long>(std::nearbyint(-min_value / initial_scale));
    const uint8_t zp = static_cast<uint8_t>(std::clamp(rounded_zp, 0L, 15L));

    const auto evaluate = [&](float scale_candidate, uint8_t* quantized) {
        const ov::float16 scale_f16(scale_candidate);
        const float runtime_scale = static_cast<float>(scale_f16);
        if (!(runtime_scale > 0.0f) || !std::isfinite(runtime_scale)) {
            return std::make_pair(
                scale_f16,
                Error{std::numeric_limits<double>::infinity(), std::numeric_limits<float>::infinity()});
        }
        for (int k = 0; k < 32; ++k) {
            const long rounded = static_cast<long>(std::nearbyint(values[k] / runtime_scale)) + zp;
            quantized[k] = static_cast<uint8_t>(std::clamp(rounded, 0L, 15L));
        }
        return std::make_pair(scale_f16, measure(runtime_scale, zp, quantized));
    };

    const auto accept = [&](const std::pair<ov::float16, Error>& candidate, const uint8_t* candidate_q) {
        const auto& [scale, error] = candidate;
        if (error.squared < best_error.squared && error.max_abs <= fallback_max_error) {
            best_scale = scale;
            best_zp = zp;
            best_error = error;
            std::copy_n(candidate_q, 32, best_q);
        }
    };

    uint8_t candidate_q[32];
    auto candidate = evaluate(initial_scale, candidate_q);
    accept(candidate, candidate_q);

    double numerator = 0.0;
    double denominator = 0.0;
    for (int k = 0; k < 32; ++k) {
        const int centered = static_cast<int>(candidate_q[k]) - zp;
        numerator += static_cast<double>(values[k]) * centered;
        denominator += static_cast<double>(centered) * centered;
    }
    if (denominator != 0.0) {
        candidate = evaluate(static_cast<float>(numerator / denominator), candidate_q);
        accept(candidate, candidate_q);
    }

    out_scale = best_scale;
    out_zp = best_zp;
    std::fill_n(out_packed, 16, 0);
    for (int k = 0; k < 32; ++k) {
        out_packed[k / 2] |= static_cast<uint8_t>(best_q[k] << (4 * (k % 2)));
    }
}

inline void unpack_gguf_32x4(const uint8_t* source, uint8_t* packed) {
    for (size_t pair = 0; pair < 8; ++pair) {
        const auto first = source[2 * pair];
        const auto second = source[2 * pair + 1];
        packed[pair] = static_cast<uint8_t>((first & 0x0f) | ((second & 0x0f) << 4));
        packed[8 + pair] = static_cast<uint8_t>((first >> 4) | (second & 0xf0));
    }
}

inline void unpack_gguf_q4_0(const uint8_t* payload, size_t byte_size, Tensor& weights, Tensor& scales) {
    constexpr size_t block_elements = 32;
    constexpr size_t block_bytes = 18;
    const auto blocks = scales.get_size();
    OPENVINO_ASSERT(byte_size % block_bytes == 0 && byte_size / block_bytes == blocks &&
                        weights.get_element_type() == element::i4 && weights.get_size() % block_elements == 0 &&
                        weights.get_size() / block_elements == blocks,
                    "Invalid GGUF Q4_0 payload or output size");
    auto* codes = static_cast<uint8_t*>(weights.data());
    auto* factors = scales.data<float16>();
    parallel_for(blocks, [&](size_t index) {
        const auto* block = payload + index * block_bytes;
        auto* packed = codes + index * (block_elements / 2);
        factors[index] = float16::from_bits(static_cast<uint16_t>(block[0] | (block[1] << 8)));
        unpack_gguf_32x4(block + 2, packed);
        for (size_t byte = 0; byte < block_elements / 2; ++byte) {
            packed[byte] ^= 0x88;
        }
    });
}

inline void unpack_gguf_q4_1_fractional(const uint8_t* payload,
                                       size_t byte_size,
                                       Tensor& weights,
                                       Tensor& scales,
                                       Tensor& zero_points) {
    constexpr size_t block_bytes = 20;
    constexpr size_t packed_bytes = 16;
    const auto blocks = scales.get_size();
    OPENVINO_ASSERT(byte_size % block_bytes == 0 && byte_size / block_bytes == blocks &&
                        weights.get_byte_size() == blocks * packed_bytes && zero_points.get_size() == blocks,
                    "Invalid GGUF Q4_1 payload or output size");
    auto* codes = static_cast<uint8_t*>(weights.data());
    auto* factors = scales.data<float16>();
    auto* offsets = zero_points.data<float16>();
    parallel_for(blocks, [&](size_t index) {
        const auto* block = payload + index * block_bytes;
        const auto scale = float16::from_bits(static_cast<uint16_t>(block[0] | (block[1] << 8)));
        const auto minimum = float16::from_bits(static_cast<uint16_t>(block[2] | (block[3] << 8)));
        const float factor = scale;
        factors[index] = float16(factor);
        offsets[index] = float16(factor != 0.f ? -static_cast<float>(minimum) / factor : 0.f);
        unpack_gguf_32x4(block + 4, codes + index * packed_bytes);
    });
}

inline void unpack_gguf_q4_1_integer(const uint8_t* payload,
                                    size_t byte_size,
                                    Tensor& weights,
                                    Tensor& scales,
                                    Tensor& zero_points) {
    constexpr size_t block_bytes = 20;
    constexpr size_t packed_bytes = 16;
    const auto blocks = scales.get_size();
    OPENVINO_ASSERT(byte_size % block_bytes == 0 && byte_size / block_bytes == blocks &&
                        weights.get_byte_size() == blocks * packed_bytes && zero_points.get_size() == blocks,
                    "Invalid GGUF Q4_1 payload or output size");
    auto* codes = static_cast<uint8_t*>(weights.data());
    auto* factors = scales.data<float16>();
    auto* offsets = zero_points.data<uint8_t>();
    parallel_for(blocks, [&](size_t index) {
        const auto* block = payload + index * block_bytes;
        const float scale = float16::from_bits(static_cast<uint16_t>(block[0] | (block[1] << 8)));
        const float minimum = float16::from_bits(static_cast<uint16_t>(block[2] | (block[3] << 8)));
        OPENVINO_ASSERT(std::isfinite(scale) && std::isfinite(minimum),
                        "[GGUF] Q4_1 block ", index, " has non-finite scale metadata");
        uint8_t source_codes[32];
        for (size_t byte = 0; byte < packed_bytes; ++byte) {
            source_codes[byte] = block[4 + byte] & 0x0f;
            source_codes[byte + packed_bytes] = block[4 + byte] >> 4;
        }
        requantize_gguf_group_u4(source_codes, scale, -minimum, factors[index], offsets[index],
                                 codes + index * packed_bytes);
    });
}

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

inline std::shared_ptr<Node> make_gguf_fc_compatible(const op::internal::FullyConnectedCompressed& fc,
                                                   const element::TypeVector& formats) {
    const auto weight = as_type_ptr<op::v0::Constant>(fc.get_input_node_shared_ptr(1));
    if (!weight || std::find(formats.begin(), formats.end(), weight->get_element_type()) == formats.end() ||
        fc.get_input_element_type(0) != element::f32 || fc.get_output_element_type(0) != element::f32) {
        return {};
    }
    const auto type = weight->get_element_type();
    if (type != element::gguf_q4_0 && type != element::gguf_q4_1 && type != element::gguf_q8_0) {
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
                    "Invalid GGUF FC weight shape: ", shape);
    const auto groups = shape[1] / group_size;
    Tensor codes(type == element::gguf_q4_1 ? element::u4 : type == element::gguf_q4_0 ? element::i4 : element::i8,
                 {shape[0], groups, group_size});
    Tensor scales(element::f16, {shape[0], groups, 1});
    const auto* payload = static_cast<const uint8_t*>(weight->get_data_ptr());
    std::shared_ptr<op::v0::Constant> zero_point_node;
    if (type == element::gguf_q4_1) {
        Tensor zero_points(element::u8, scales.get_shape());
        unpack_gguf_q4_1_integer(payload, weight->get_byte_size(), codes, scales, zero_points);
        zero_point_node = std::make_shared<op::v0::Constant>(zero_points);
    } else {
        const auto unpack = type == element::gguf_q4_0 ? unpack_gguf_q4_0 : unpack_gguf_q8_0;
        unpack(payload, weight->get_byte_size(), codes, scales);
    }
    const auto code_node = std::make_shared<op::v0::Constant>(codes);
    const auto scale_node = std::make_shared<op::v0::Constant>(scales);
    const auto target = std::make_shared<op::v0::Constant>(element::i64, Shape{shape.size()}, shape);
    const auto dequantized = decomposition::low_precision_dequantize(code_node, scale_node, zero_point_node, target);
    const auto converted = std::make_shared<op::v0::Convert>(dequantized, element::f32);
    return std::make_shared<op::v0::MatMul>(fc.input_value(0), converted, false, true);
}

inline std::shared_ptr<Node> make_gguf_q8_0_fc_compatible(const op::internal::FullyConnectedCompressed& fc) {
    return make_gguf_fc_compatible(fc, {element::gguf_q8_0});
}

inline void lower_gguf_fc(Model& model, const element::TypeVector& formats) {
    for (const auto& node : model.get_ordered_ops()) {
        const auto fc = as_type_ptr<op::internal::FullyConnectedCompressed>(node);
        if (!fc) {
            continue;
        }
        if (const auto compatible = make_gguf_fc_compatible(*fc, formats)) {
            compatible->set_friendly_name(node->get_friendly_name());
            copy_runtime_info(node, compatible);
            replace_node(node, compatible);
        }
    }
}

inline void lower_gguf_q8_0_fc(Model& model) {
    lower_gguf_fc(model, {element::gguf_q8_0});
}

}  // namespace ov::util