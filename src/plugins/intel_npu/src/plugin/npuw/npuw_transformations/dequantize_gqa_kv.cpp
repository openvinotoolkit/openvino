// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dequantize_gqa_kv.hpp"

#include <memory>
#include <string>
#include <vector>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_left_shift.hpp"
#include "openvino/op/bitwise_or.hpp"
#include "openvino/op/bitwise_right_shift.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/group_query_attention.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"

namespace {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v5 = ov::op::v5;
namespace v13 = ov::op::v13;
namespace v15 = ov::op::v15;
using ov::op::internal::GroupQueryAttention;

// Reshape the flat per-(layer,role) scale into the KV cache layout [1, kv_num_heads, 1, head_size]
// (PER_CHANNEL) or [1,1,1,1] (PER_TENSOR). When the scale is a Constant, build a FRESH single-reader
// Constant that owns a memcpy'd copy of the data (data-copy ctor) so each (layer,K/V) scale has a unique
// pointer: NPUW's FOLD pass needs that to build a complete per-instance scale bank (exporters reuse one
// scale node across identical layers; a Reshape/aliased clone gets deduped and breaks weight-bank matching).
std::shared_ptr<ov::Node> make_kv_scale(const ov::Output<ov::Node>& scale,
                                        int64_t kv_num_heads,
                                        const std::string& quant_type) {
    const bool per_channel = (quant_type == "PER_CHANNEL");
    int64_t head_size = 1;
    if (per_channel) {
        const auto& ps = scale.get_partial_shape();
        if (ps.rank().is_static() && ps.rank().get_length() == 1 && ps[0].is_static()) {
            head_size = ps[0].get_length() / kv_num_heads;
        } else {
            head_size = -1;
        }
    }
    const ov::Shape new_shape{1,
                              static_cast<size_t>(per_channel ? kv_num_heads : 1),
                              1,
                              static_cast<size_t>(per_channel ? head_size : 1)};

    if (const auto scale_const = ov::as_type_ptr<v0::Constant>(scale.get_node_shared_ptr())) {
        if (!per_channel || head_size > 0) {
            return std::make_shared<v0::Constant>(scale_const->get_element_type(),
                                                  new_shape,
                                                  scale_const->get_data_ptr());
        }
    }
    std::vector<int64_t> target = {1, per_channel ? kv_num_heads : 1, 1, per_channel ? head_size : 1};
    const auto shape_const = v0::Constant::create(ov::element::i64, ov::Shape{target.size()}, target);
    return std::make_shared<v1::Reshape>(scale, shape_const, false);
}

// Symmetric dequantization to the compute type, applied to the isolated KV inputs only.
//  - 8-bit: Convert(i8 -> compute) * scale.
//  - 4-bit: u8 stores two signed values per byte (low nibble = even channel, high = odd) biased by +8.
//    Unpack nibbles -> interleave back along head_size -> (Convert - 8) * scale. Mirrors the common
//    GroupQueryAttentionDecomposition::dequantize_kv 4-bit path.
ov::Output<ov::Node> dequantize_kv(const ov::Output<ov::Node>& quantized,
                                   const ov::Output<ov::Node>& scale,
                                   int64_t kv_num_heads,
                                   int64_t kv_cache_bit_width,
                                   const std::string& quant_type,
                                   const ov::element::Type& compute_type) {
    auto scale4d = make_kv_scale(scale, kv_num_heads, quant_type);
    ov::Output<ov::Node> scale_ct = scale4d;
    if (scale.get_element_type() != compute_type) {
        scale_ct = std::make_shared<v0::Convert>(scale4d, compute_type);
    }

    if (kv_cache_bit_width == 4) {
        const auto qtype = quantized.get_element_type();
        const auto axis_last = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        const auto mask_low = v0::Constant::create(qtype, ov::Shape{}, {0x0F});
        const auto shift_4 = v0::Constant::create(qtype, ov::Shape{}, {4});
        const auto low_nibble = std::make_shared<v13::BitwiseAnd>(quantized, mask_low);
        const auto high_nibble = std::make_shared<v15::BitwiseRightShift>(quantized, shift_4);
        const auto low_u = std::make_shared<v0::Unsqueeze>(low_nibble, axis_last);
        const auto high_u = std::make_shared<v0::Unsqueeze>(high_nibble, axis_last);
        const auto interleaved = std::make_shared<v0::Concat>(ov::OutputVector{low_u, high_u}, -1);
        const auto flat_shape = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 0, 0, -1});
        const auto unpacked = std::make_shared<v1::Reshape>(interleaved, flat_shape, true);
        const auto zp = std::make_shared<v0::Convert>(v0::Constant::create(qtype, ov::Shape{}, {8}), compute_type);
        const auto conv = std::make_shared<v0::Convert>(unpacked, compute_type);
        const auto unbiased = std::make_shared<v1::Subtract>(conv, zp);
        return std::make_shared<v1::Multiply>(unbiased, scale_ct);
    }

    const auto converted = std::make_shared<v0::Convert>(quantized, compute_type);
    return std::make_shared<v1::Multiply>(converted, scale_ct);
}

// Symmetric quantize-on-write: clamp(round(x * inv_scale)) -> cache_type, with MLAS SafeInvScale zero guard.
//  - 8-bit: Clamp[-128,127] -> Convert(i8).
//  - 4-bit: Clamp[-8,7] -> +8 bias -> Convert(u8) -> pack channel pairs into nibbles (low|high<<4).
//    Mirrors the common GroupQueryAttentionDecomposition::quantize_kv 4-bit path.
ov::Output<ov::Node> quantize_kv(const ov::Output<ov::Node>& current,
                                 const ov::Output<ov::Node>& scale,
                                 int64_t kv_num_heads,
                                 int64_t kv_cache_bit_width,
                                 const std::string& quant_type,
                                 const ov::element::Type& cache_type) {
    const auto compute_type = current.get_element_type();
    auto scale4d = make_kv_scale(scale, kv_num_heads, quant_type);
    ov::Output<ov::Node> scale_ct = scale4d;
    if (scale.get_element_type() != compute_type) {
        scale_ct = std::make_shared<v0::Convert>(scale4d, compute_type);
    }
    const auto one = v0::Constant::create(compute_type, ov::Shape{}, {1});
    const auto zero = v0::Constant::create(compute_type, ov::Shape{}, {0});
    const auto is_zero = std::make_shared<v1::Equal>(scale_ct, zero);
    const auto safe = std::make_shared<v1::Select>(is_zero, one, scale_ct);
    const auto inv = std::make_shared<v1::Divide>(one, safe);
    const auto scaled = std::make_shared<v1::Multiply>(current, inv);
    const auto rounded = std::make_shared<v5::Round>(scaled, v5::Round::RoundMode::HALF_TO_EVEN);

    if (kv_cache_bit_width == 4) {
        const auto clamped = std::make_shared<v0::Clamp>(rounded, -8.0, 7.0);
        const auto bias = v0::Constant::create(compute_type, ov::Shape{}, {8});
        const auto biased = std::make_shared<v1::Add>(clamped, bias);
        const auto as_u8 = std::make_shared<v0::Convert>(biased, cache_type);
        const auto pair_shape = v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, -1, 2});
        const auto paired = std::make_shared<v1::Reshape>(as_u8, pair_shape, true);
        const auto axis_last = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        const auto split =
            std::make_shared<v1::VariadicSplit>(paired,
                                                v0::Constant::create(ov::element::i64, ov::Shape{}, {-1}),
                                                v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1}));
        const auto low = std::make_shared<v0::Squeeze>(split->output(0), axis_last);
        const auto high = std::make_shared<v0::Squeeze>(split->output(1), axis_last);
        const auto shift_4 = v0::Constant::create(cache_type, ov::Shape{}, {4});
        const auto mask_low = v0::Constant::create(cache_type, ov::Shape{}, {0x0F});
        const auto low_masked = std::make_shared<v13::BitwiseAnd>(low, mask_low);
        const auto high_shifted = std::make_shared<v15::BitwiseLeftShift>(high, shift_4);
        return std::make_shared<v13::BitwiseOr>(low_masked, high_shifted);
    }

    const auto clamped = std::make_shared<v0::Clamp>(rounded, -128.0, 127.0);
    return std::make_shared<v0::Convert>(clamped, cache_type);
}

}  // namespace

bool ov::npuw::DequantizeGQAKVCache::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool changed = false;
    // Collect first so we can safely replace while iterating.
    std::vector<std::shared_ptr<GroupQueryAttention>> targets;
    for (const auto& op : model->get_ordered_ops()) {
        auto gqa = ov::as_type_ptr<GroupQueryAttention>(op);
        if (gqa && gqa->is_kv_quantized() &&
            (gqa->get_kv_cache_bit_width() == 8 || gqa->get_kv_cache_bit_width() == 4)) {
            targets.push_back(gqa);
        }
    }

    constexpr size_t PAST_KEY = 3, PAST_VALUE = 4, K_SCALE = 12, V_SCALE = 13;
    for (const auto& node : targets) {
        const auto& args = node->input_values();
        if (args.size() <= V_SCALE) {
            continue;
        }
        const int64_t kv_num_heads = node->get_kv_num_heads();
        const int64_t bit_width = node->get_kv_cache_bit_width();
        const auto compute_type = node->get_input_element_type(0);  // Q precision = attention math type
        const auto cache_type = node->get_input_element_type(PAST_KEY);
        const std::string k_qt = node->get_k_quant_type();
        const std::string v_qt = node->get_v_quant_type();
        const auto k_scale = args[K_SCALE];
        const auto v_scale = args[V_SCALE];

        // Dequantize the two separate KV inputs (never the packed QKV at input 0).
        const auto past_key_f = dequantize_kv(args[PAST_KEY], k_scale, kv_num_heads, bit_width, k_qt, compute_type);
        const auto past_value_f = dequantize_kv(args[PAST_VALUE], v_scale, kv_num_heads, bit_width, v_qt, compute_type);

        // Rebuild the op float: keep inputs [0, 12), swap dequantized KV, drop scale inputs, strip trailing
        // NullNode placeholders down to the 7 mandatory inputs (unused optional slots serialize as an
        // unsupported "extension" op otherwise), and use the plain float constructor (no quant metadata).
        ov::OutputVector new_args(args.begin(), args.begin() + K_SCALE);
        new_args[PAST_KEY] = past_key_f;
        new_args[PAST_VALUE] = past_value_f;
        while (new_args.size() > 7 && new_args.back().get_node_shared_ptr()->description() == "NullNode") {
            new_args.pop_back();
        }
        auto float_gqa = std::make_shared<GroupQueryAttention>(new_args,
                                                               node->get_num_heads(),
                                                               node->get_kv_num_heads(),
                                                               node->get_scale(),
                                                               node->get_do_rotary(),
                                                               node->get_rotary_interleaved());
        float_gqa->set_friendly_name(node->get_friendly_name());

        // Requantize the float present KV back to the packed cache type so model outputs keep the i8/i4 contract.
        const auto present_key_i8 =
            quantize_kv(float_gqa->output(1), k_scale, kv_num_heads, bit_width, k_qt, cache_type);
        const auto present_value_i8 =
            quantize_kv(float_gqa->output(2), v_scale, kv_num_heads, bit_width, v_qt, cache_type);

        node->output(0).replace(float_gqa->output(0));
        node->output(1).replace(present_key_i8);
        node->output(2).replace(present_value_i8);
        ov::copy_runtime_info(node, float_gqa);
        changed = true;
    }
    return changed;
}
