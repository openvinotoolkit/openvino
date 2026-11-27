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
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/tensor.hpp"
#include "ov_ops/fully_connected_compressed.hpp"
#include "ov_ops/gather_compressed.hpp"
#include "ov_ops/gather_matmul.hpp"

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

inline void get_gguf_k_scale_min(int group, const uint8_t* packed, uint8_t* scale, uint8_t* minimum) {
    if (group < 4) {
        *scale = packed[group] & 63;
        *minimum = packed[group + 4] & 63;
    } else {
        *scale = (packed[group + 4] & 15) | ((packed[group - 4] >> 6) << 4);
        *minimum = (packed[group + 4] >> 4) | ((packed[group] >> 6) << 4);
    }
}

inline void unpack_gguf_q4_k(const uint8_t* payload,
                             size_t byte_size,
                             Tensor& weights,
                             Tensor& scales,
                             Tensor& zero_points) {
    constexpr size_t block_bytes = 144;
    constexpr size_t packed_bytes = 128;
    constexpr size_t groups_per_block = 8;
    const auto blocks = byte_size / block_bytes;
    OPENVINO_ASSERT(byte_size % block_bytes == 0 && weights.get_byte_size() == blocks * packed_bytes &&
                        scales.get_size() == blocks * groups_per_block && zero_points.get_size() == scales.get_size(),
                    "Invalid GGUF Q4_K payload or output size");
    auto* codes = static_cast<uint8_t*>(weights.data());
    auto* factors = scales.data<float16>();
    const bool integer = zero_points.get_element_type() == element::u8;
    auto* integer_zp = integer ? zero_points.data<uint8_t>() : nullptr;
    auto* fractional_zp = integer ? nullptr : zero_points.data<float16>();
    parallel_for(blocks, [&](size_t index) {
        const auto* block = payload + index * block_bytes;
        const float block_scale = float16::from_bits(static_cast<uint16_t>(block[0] | (block[1] << 8)));
        const float block_min = float16::from_bits(static_cast<uint16_t>(block[2] | (block[3] << 8)));
        OPENVINO_ASSERT(std::isfinite(block_scale) && std::isfinite(block_min),
                        "[GGUF] Q4_K block ", index, " has non-finite scale metadata");
        for (int group = 0; group < 8; ++group) {
            uint8_t scale_code, min_code;
            get_gguf_k_scale_min(group, block + 4, &scale_code, &min_code);
            const float scale = block_scale * scale_code;
            const float minimum = block_min * min_code;
            const auto group_index = index * groups_per_block + group;
            const auto* source = block + 16 + (group / 2) * 32;
            auto* packed = codes + group_index * 16;
            const auto shift = (group % 2) * 4;
            if (integer) {
                uint8_t source_codes[32];
                for (size_t lane = 0; lane < 32; ++lane) {
                    source_codes[lane] = (source[lane] >> shift) & 15;
                }
                requantize_gguf_group_u4(source_codes, scale, minimum, factors[group_index],
                                         integer_zp[group_index], packed);
            } else {
                factors[group_index] = float16(scale);
                fractional_zp[group_index] = float16(scale != 0.f ? minimum / scale : 0.f);
                for (size_t pair = 0; pair < 16; ++pair) {
                    packed[pair] = static_cast<uint8_t>(((source[2 * pair] >> shift) & 15) |
                                                       (((source[2 * pair + 1] >> shift) & 15) << 4));
                }
            }
        }
    });
}

inline void dequantize_gguf_q5_k_block(const uint8_t* block, float* values) {
    const float scale = float16::from_bits(static_cast<uint16_t>(block[0] | (block[1] << 8)));
    const float minimum = float16::from_bits(static_cast<uint16_t>(block[2] | (block[3] << 8)));
    for (int group = 0; group < 8; ++group) {
        uint8_t scale_code, min_code;
        get_gguf_k_scale_min(group, block + 4, &scale_code, &min_code);
        const float group_scale = scale * scale_code;
        const float group_min = minimum * min_code;
        const auto* low = block + 48 + (group / 2) * 32;
        for (size_t lane = 0; lane < 32; ++lane) {
            const auto code = ((low[lane] >> ((group % 2) * 4)) & 15) | (((block[16 + lane] >> group) & 1) << 4);
            values[group * 32 + lane] = group_scale * static_cast<float>(code) - group_min;
        }
    }
}

inline void unpack_gguf_q5_k_f16_zp(const uint8_t* payload,
                                   size_t byte_size,
                                   Tensor& weights,
                                   Tensor& scales,
                                   Tensor& zero_points) {
    constexpr size_t block_bytes = 176;
    constexpr size_t groups_per_block = 8;
    constexpr size_t group_size = 32;
    const auto blocks = byte_size / block_bytes;
    OPENVINO_ASSERT(byte_size % block_bytes == 0 && scales.get_size() == blocks * groups_per_block &&
                        zero_points.get_size() == scales.get_size() && weights.get_size() == scales.get_size() * group_size,
                    "Invalid GGUF Q5_K payload or output size");
    auto* codes = weights.data<uint8_t>();
    auto* factors = scales.data<float16>();
    auto* offsets = zero_points.data<float16>();
    parallel_for(blocks, [&](size_t index) {
        const auto* block = payload + index * block_bytes;
        const float block_scale = float16::from_bits(static_cast<uint16_t>(block[0] | (block[1] << 8)));
        const float block_min = float16::from_bits(static_cast<uint16_t>(block[2] | (block[3] << 8)));
        for (int group = 0; group < 8; ++group) {
            uint8_t scale_code, min_code;
            get_gguf_k_scale_min(group, block + 4, &scale_code, &min_code);
            const float scale = block_scale * scale_code;
            const float minimum = block_min * min_code;
            const auto offset = index * groups_per_block + group;
            factors[offset] = float16(scale);
            offsets[offset] = float16(scale != 0.f ? minimum / scale : 0.f);
            const auto* low = block + 48 + (group / 2) * group_size;
            for (size_t lane = 0; lane < group_size; ++lane) {
                codes[offset * group_size + lane] = static_cast<uint8_t>(
                    ((low[lane] >> ((group % 2) * 4)) & 15) | (((block[16 + lane] >> group) & 1) << 4));
            }
        }
    });
}

inline void unpack_gguf_q5_k_i8(const uint8_t* payload, size_t byte_size, Tensor& weights, Tensor& scales) {
    constexpr size_t block_bytes = 176;
    constexpr size_t block_elements = 256;
    constexpr size_t group_size = 32;
    const auto blocks = byte_size / block_bytes;
    OPENVINO_ASSERT(byte_size % block_bytes == 0 && weights.get_size() % block_elements == 0 &&
                        weights.get_size() / block_elements == blocks && scales.get_size() == weights.get_size() / group_size,
                    "[GGUF] invalid Q5_K symmetric requantization payload or buffers");
    auto* codes = weights.data<int8_t>();
    auto* factors = scales.data<float16>();
    parallel_for(blocks, [&](size_t block) {
        float values[block_elements];
        dequantize_gguf_q5_k_block(payload + block * block_bytes, values);
        for (size_t offset = 0; offset < block_elements; offset += group_size) {
            float amax = 0.f;
            for (size_t lane = offset; lane < offset + group_size; ++lane) {
                OPENVINO_ASSERT(std::isfinite(values[lane]), "[GGUF] cannot requantize non-finite Q5_K values");
                amax = std::max(amax, std::fabs(values[lane]));
            }
            const float16 stored_scale(amax / 127.f);
            const float scale = stored_scale;
            OPENVINO_ASSERT(std::isfinite(scale), "[GGUF] Q5_K symmetric scale exceeds f16 range");
            factors[(block * block_elements + offset) / group_size] = stored_scale;
            for (size_t lane = offset; lane < offset + group_size; ++lane) {
                const float code = scale > 0.f ? std::round(values[lane] / scale) : 0.f;
                codes[block * block_elements + lane] = static_cast<int8_t>(std::clamp(code, -127.f, 127.f));
            }
        }
    });
}

inline void unpack_gguf_q6_k(const uint8_t* payload, size_t byte_size, Tensor& weights, Tensor& scales) {
    constexpr size_t block_bytes = 210;
    constexpr size_t block_elements = 256;
    constexpr size_t groups_per_block = 16;
    const auto blocks = byte_size / block_bytes;
    OPENVINO_ASSERT(byte_size % block_bytes == 0 && weights.get_size() % block_elements == 0 &&
                        weights.get_size() / block_elements == blocks && scales.get_size() == blocks * groups_per_block,
                    "Invalid GGUF Q6_K payload or output size");
    auto* codes = weights.data<int8_t>();
    auto* factors = scales.data<float16>();
    parallel_for(blocks, [&](size_t index) {
        const auto* block = payload + index * block_bytes;
        const float scale = float16::from_bits(static_cast<uint16_t>(block[208] | (block[209] << 8)));
        const auto* subscales = reinterpret_cast<const int8_t*>(block + 192);
        for (size_t group = 0; group < groups_per_block; ++group) {
            factors[index * groups_per_block + group] = float16(scale * subscales[group]);
        }
        for (size_t half = 0; half < 2; ++half) {
            const auto* low = block + half * 64;
            const auto* high = block + 128 + half * 32;
            auto* out = codes + index * block_elements + half * 128;
            for (size_t lane = 0; lane < 32; ++lane) {
                out[lane] = static_cast<int8_t>(((low[lane] & 15) | ((high[lane] & 3) << 4)) - 32);
                out[lane + 32] = static_cast<int8_t>(((low[lane + 32] & 15) | (((high[lane] >> 2) & 3) << 4)) - 32);
                out[lane + 64] = static_cast<int8_t>(((low[lane] >> 4) | (((high[lane] >> 4) & 3) << 4)) - 32);
                out[lane + 96] = static_cast<int8_t>(((low[lane + 32] >> 4) | ((high[lane] >> 6) << 4)) - 32);
            }
        }
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

inline void lower_gguf_q8_0_experts(Model& model) {
    for (const auto& node : model.get_ordered_ops()) {
        const auto matmul = as_type_ptr<op::internal::GatherMatmul>(node);
        if (!matmul || matmul->get_input_element_type(1) != element::gguf_q8_0)
            continue;
        const auto weights = as_type_ptr<op::v0::Constant>(matmul->get_input_node_shared_ptr(1));
        if (!weights)
            continue;
        const auto& shape = weights->get_shape();
        constexpr size_t group_size = 32;
        if (shape.size() != 3 || !shape[0] || !shape[1] || !shape[2] || shape[2] % group_size)
            continue;
        const auto groups = shape[2] / group_size;
        Tensor codes(element::i8, {shape[0], shape[1], groups, group_size});
        Tensor scales(element::f16, {shape[0], shape[1], groups, 1});
        unpack_gguf_q8_0(static_cast<const uint8_t*>(weights->get_data_ptr()), weights->get_byte_size(), codes, scales);
        const auto code_node = std::make_shared<op::v0::Constant>(codes);
        const auto scale_node = std::make_shared<op::v0::Constant>(scales);
        const auto target = op::v0::Constant::create(element::i64, {shape.size()}, shape);
        const auto decoded = decomposition::low_precision_dequantize(code_node, scale_node, {}, target);
        const auto converted = std::make_shared<op::v0::Convert>(decoded, element::f32);
        matmul->input(1).replace_source_output(converted);
    }
}

inline void lower_gguf_vocab_projection(Model& model) {
    if (model.get_results().empty())
        return;
    auto output = model.get_results().front()->input_value(0);
    for (size_t depth = 0; depth < 6; ++depth) {
        const auto node = output.get_node_shared_ptr();
        if (is_type<op::v1::Reshape>(node) || is_type<op::v0::Unsqueeze>(node) || is_type<op::v0::Tanh>(node)) {
            output = node->input_value(0);
        } else if (is_type<op::v1::Multiply>(node) || is_type<op::v1::Divide>(node)) {
            const auto scalar = as_type_ptr<op::v0::Constant>(node->get_input_node_shared_ptr(1));
            if (!scalar || shape_size(scalar->get_shape()) != 1)
                return;
            output = node->input_value(0);
        } else {
            break;
        }
    }
    const auto head = as_type_ptr<op::internal::FullyConnectedCompressed>(output.get_node_shared_ptr());
    if (!head || head->get_input_element_type(0) != element::f32 || head->get_element_type() != element::f32)
        return;
    const auto weight = as_type_ptr<op::v0::Constant>(head->get_input_node_shared_ptr(1));
    if (!weight)
        return;
    const auto type = weight->get_element_type();
    const auto& shape = weight->get_shape();
    if ((type != element::gguf_q4_k && type != element::gguf_q5_k) || shape.size() != 2 || !shape[0] || !shape[1] ||
        shape[1] % type.block_elem_count() != 0)
        return;
    for (size_t port = 2; port < head->get_input_size(); ++port) {
        const auto absent = as_type_ptr<op::v0::Constant>(head->get_input_node_shared_ptr(port));
        if (!absent || shape_size(absent->get_shape()) != 0)
            return;
    }
    const auto ops = model.get_ordered_ops();
    const bool has_token_embedding = std::any_of(ops.begin(), ops.end(), [&](const auto& node) {
        if ((!is_type<op::v8::Gather>(node) && !is_type<op::internal::GatherCompressed>(node)) ||
            !node->get_input_partial_shape(0).is_static() || node->get_input_shape(0) != shape)
            return false;
        const auto axis = as_type_ptr<op::v0::Constant>(node->get_input_node_shared_ptr(2));
        if (!axis || axis->cast_vector<int64_t>() != std::vector<int64_t>{0})
            return false;
        auto indices = node->get_input_node_shared_ptr(1);
        while (is_type<op::v0::Squeeze>(indices) || is_type<op::v0::Unsqueeze>(indices) ||
               is_type<op::v1::Reshape>(indices) || is_type<op::v0::Convert>(indices)) {
            indices = indices->get_input_node_shared_ptr(0);
        }
        return is_type<op::v0::Parameter>(indices) && indices->get_element_type().is_integral_number();
    });
    if (!has_token_embedding)
        return;

    const bool q4_k = type == element::gguf_q4_k;
    const size_t groups = shape[1] / 32;
    Tensor codes(q4_k ? element::u4 : element::u8, {shape[0], groups, 32});
    Tensor scales(element::f16, {shape[0], groups, 1});
    Tensor zero_points(element::f16, scales.get_shape());
    const auto unpack = q4_k ? unpack_gguf_q4_k : unpack_gguf_q5_k_f16_zp;
    unpack(weight->get_data_ptr<uint8_t>(), weight->get_byte_size(), codes, scales, zero_points);
    const auto as_f32 = [](const Tensor& tensor) {
        return std::make_shared<op::v0::Convert>(std::make_shared<op::v0::Constant>(tensor), element::f32);
    };
    const auto centered = std::make_shared<op::v1::Subtract>(as_f32(codes), as_f32(zero_points));
    const auto scaled = std::make_shared<op::v1::Multiply>(centered, as_f32(scales));
    const auto matrix = std::make_shared<op::v1::Reshape>(
        scaled, op::v0::Constant::create(element::i64, {2}, shape), false);
    const auto replacement = std::make_shared<op::v0::MatMul>(head->input_value(0), matrix, false, true);
    copy_runtime_info(head, replacement);
    replace_node_update_name(head, replacement);
}

inline std::shared_ptr<Node> make_gguf_fc_compatible(const op::internal::FullyConnectedCompressed& fc,
                                                   const element::TypeVector& formats) {
    const auto weight = as_type_ptr<op::v0::Constant>(fc.get_input_node_shared_ptr(1));
    if (!weight || std::find(formats.begin(), formats.end(), weight->get_element_type()) == formats.end() ||
        fc.get_input_element_type(0) != element::f32 || fc.get_output_element_type(0) != element::f32) {
        return {};
    }
    const auto type = weight->get_element_type();
    if (type != element::gguf_q4_0 && type != element::gguf_q4_1 && type != element::gguf_q8_0 &&
        type != element::gguf_q6_k && type != element::gguf_q4_k && type != element::gguf_q5_k) {
        return {};
    }
    for (size_t port = 2; port < fc.get_input_size(); ++port) {
        const auto absent = as_type_ptr<op::v0::Constant>(fc.get_input_node_shared_ptr(port));
        if (!absent || shape_size(absent->get_shape()) != 0) {
            return {};
        }
    }
    const auto& shape = weight->get_shape();
    const bool q6_k = type == element::gguf_q6_k;
    const bool asymmetric = type == element::gguf_q4_1 || type == element::gguf_q4_k;
    const size_t group_size = q6_k ? 16 : 32;
    const size_t block_size = weight->get_element_type().block_elem_count();
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] && shape[1] && shape[1] % block_size == 0,
                    "Invalid GGUF FC weight shape: ", shape);
    const auto groups = shape[1] / group_size;
    Tensor codes(asymmetric ? element::u4 : type == element::gguf_q4_0 ? element::i4 : element::i8,
                 {shape[0], groups, group_size});
    Tensor scales(element::f16, {shape[0], groups, 1});
    const auto* payload = static_cast<const uint8_t*>(weight->get_data_ptr());
    std::shared_ptr<op::v0::Constant> zero_point_node;
    if (asymmetric) {
        Tensor zero_points(element::u8, scales.get_shape());
        const auto unpack = type == element::gguf_q4_k ? unpack_gguf_q4_k : unpack_gguf_q4_1_integer;
        unpack(payload, weight->get_byte_size(), codes, scales, zero_points);
        zero_point_node = std::make_shared<op::v0::Constant>(zero_points);
    } else if (q6_k) {
        unpack_gguf_q6_k(payload, weight->get_byte_size(), codes, scales);
    } else if (type == element::gguf_q5_k) {
        unpack_gguf_q5_k_i8(payload, weight->get_byte_size(), codes, scales);
    } else {
        const auto unpack = type == element::gguf_q4_0 ? unpack_gguf_q4_0 : unpack_gguf_q8_0;
        unpack(payload, weight->get_byte_size(), codes, scales);
    }
    const auto code_node = std::make_shared<op::v0::Constant>(codes);
    const auto scale_node = std::make_shared<op::v0::Constant>(scales);
    const auto target = std::make_shared<op::v0::Constant>(element::i64, Shape{shape.size()}, shape);
    if (q6_k) {
        const auto codes_f32 = std::make_shared<op::v0::Convert>(code_node, element::f32);
        const auto scales_f32 = std::make_shared<op::v0::Convert>(scale_node, element::f32);
        const auto product = std::make_shared<op::v1::Multiply>(codes_f32, scales_f32);
        const auto reshaped = std::make_shared<op::v1::Reshape>(product, target, false);
        return std::make_shared<op::v0::MatMul>(fc.input_value(0), reshaped, false, true);
    }
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